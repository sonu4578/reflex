"""Tests for development-time event-loop blocking detection."""

import asyncio
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Coroutine, Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reflex_base.event.processor import blocking
from reflex_base.event.processor.blocking import (
    EVENT_TASK_PREFIX,
    EventLoopBlockingMonitor,
)
from reflex_base.registry import RegisteredEventHandler

from reflex.event import EventHandler

# Seconds the blocking handlers hold the loop, well past the monitor threshold.
_BLOCK_SECONDS = 0.12


def _blocked_total(text: str, subject: str) -> float:
    """Extract the total blocking duration the monitor reported for a subject.

    Args:
        text: The captured log text.
        subject: The subject of the report, such as ``Task 'name'``.

    Returns:
        The reported duration in seconds.
    """
    match = re.search(
        rf"{re.escape(subject)} blocked the event loop for (\d+\.\d+)s in total\.",
        text,
    )
    assert match is not None, text
    return float(match[1])


def _track_queued_probe(
    loop: asyncio.AbstractEventLoop, monkeypatch: pytest.MonkeyPatch
) -> threading.Event:
    """Return an event set after the monitor queues a responsiveness probe."""
    probe_queued = threading.Event()
    call_soon_threadsafe = loop.call_soon_threadsafe

    def track_probe(
        callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> asyncio.Handle:
        """Queue the callback and record that a probe was queued.

        Args:
            callback: The callback to queue.
            *args: Positional arguments for the callback.
            **kwargs: Keyword arguments for call_soon_threadsafe.

        Returns:
            The handle of the queued callback.
        """
        handle = call_soon_threadsafe(callback, *args, **kwargs)
        probe_queued.set()
        return handle

    monkeypatch.setattr(loop, "call_soon_threadsafe", track_probe)
    return probe_queued


async def _run_monitored(
    fn: Callable[[], Coroutine[Any, Any, None]],
    task_name: str,
    event_handlers: dict[str, RegisteredEventHandler] | None = None,
    app_root: str | None = None,
) -> None:
    """Run fn in a task with the given name while a monitor watches the loop.

    Args:
        fn: The coroutine function to run.
        task_name: The name of the task that runs fn.
        event_handlers: The registered handlers, by event name.
        app_root: The app root to give the monitor, instead of the working directory.
    """
    monitor = EventLoopBlockingMonitor(
        asyncio.get_running_loop(), threshold=0.04, event_handlers=event_handlers or {}
    )
    if app_root is not None:
        monitor._app_root = app_root
    monitor.start()
    try:
        await asyncio.create_task(fn(), name=task_name)
        await asyncio.sleep(0)
    finally:
        monitor.stop()


def _blocking(
    probe_queued: threading.Event, duration: float
) -> Callable[[], Coroutine[Any, Any, None]]:
    """Create a coroutine function that blocks the loop after a probe is queued.

    Returns:
        The blocking coroutine function.
    """

    async def blocking_handler():  # noqa: RUF029
        """Block the loop for the duration after a probe is queued."""
        assert probe_queued.wait(timeout=1)
        time.sleep(duration)  # noqa: ASYNC251

    return blocking_handler


@pytest.fixture(autouse=True)
def reset_reported_library_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget the library handlers an earlier test reported."""
    monkeypatch.setattr(blocking, "_reported_library_handlers", set(), raising=False)


@pytest.fixture
def caplog_blocking(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture warnings of the blocking monitor.

    Returns:
        The log capture fixture.
    """
    caplog.set_level(logging.WARNING, logger="reflex_base.event.processor.blocking")
    return caplog


@pytest.fixture
def warned() -> Generator[threading.Event, None, None]:
    """Yield an event that is set once the blocking monitor logs a warning."""
    event = threading.Event()

    class _SetOnWarning(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            """Record that the monitor logged a warning.

            Args:
                record: The log record.
            """
            event.set()

    handler = _SetOnWarning(level=logging.WARNING)
    blocking_logger = logging.getLogger("reflex_base.event.processor.blocking")
    blocking_logger.addHandler(handler)
    try:
        yield event
    finally:
        blocking_logger.removeHandler(handler)


async def test_blocking_event_is_reported_with_handler_name(
    caplog_blocking: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """Name the registered handler of the blocking event task."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)
    handler = _blocking(probe_queued, _BLOCK_SECONDS)
    registered = RegisteredEventHandler(handler=EventHandler(fn=handler), states=())

    await _run_monitored(
        handler,
        f"{EVENT_TASK_PREFIX}state.blocking|token|txid",
        {"state.blocking": registered},
    )

    assert "blocking_handler" in caplog_blocking.text
    # The total covers the whole block, not just the time until it was detected.
    # A loaded runner only lengthens it, so there is no upper bound.
    assert (
        _blocked_total(
            caplog_blocking.text,
            "Event handler _blocking.<locals>.blocking_handler",
        )
        >= _BLOCK_SECONDS
    )
    assert "time.sleep(duration)  # noqa: ASYNC251" in caplog_blocking.text
    assert "_run_monitored" not in caplog_blocking.text
    assert "thread pool or use a non-blocking async API" in caplog_blocking.text


async def test_unregistered_event_is_reported_with_event_name(
    caplog_blocking: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """Fall back to the event name when no handler is registered for it."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)

    await _run_monitored(
        _blocking(probe_queued, _BLOCK_SECONDS),
        f"{EVENT_TASK_PREFIX}state.unknown|token|txid",
    )

    assert "Event handler state.unknown blocked" in caplog_blocking.text


async def test_blocking_just_over_threshold_is_detected(
    caplog_blocking: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """Account for the time between responsiveness probes."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)

    await _run_monitored(
        _blocking(probe_queued, 0.055), f"{EVENT_TASK_PREFIX}state.blocking|token|txid"
    )

    assert "state.blocking" in caplog_blocking.text


async def test_blocking_outside_event_task_is_reported_with_task_name(
    caplog_blocking: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """Report blocking in a task that processes no event by the task name."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)

    await _run_monitored(_blocking(probe_queued, _BLOCK_SECONDS), "other_task")

    assert "Task 'other_task' is blocking the event loop" in caplog_blocking.text
    assert _blocked_total(caplog_blocking.text, "Task 'other_task'") >= _BLOCK_SECONDS
    assert "time.sleep(duration)  # noqa: ASYNC251" in caplog_blocking.text
    assert "Event handler" not in caplog_blocking.text
    assert "computed vars" not in caplog_blocking.text


async def test_awaiting_event_is_not_reported(
    caplog_blocking: pytest.LogCaptureFixture,
):
    """Do not report handlers that yield control while doing async work."""

    async def yielding_handler():
        """Yield to the loop while waiting."""
        await asyncio.sleep(0.1)

    await _run_monitored(
        yielding_handler, f"{EVENT_TASK_PREFIX}state.yielding|token|txid"
    )

    assert "the event loop" not in caplog_blocking.text


async def test_hung_event_is_reported_while_still_blocked(
    caplog_blocking: pytest.LogCaptureFixture, warned: threading.Event
):
    """Warn when the threshold is reached, before the loop responds again."""

    async def hung_handler():  # noqa: RUF029
        """Hold the loop until the monitor warns, like a handler that never returns."""
        assert warned.wait(timeout=1)

    await _run_monitored(hung_handler, f"{EVENT_TASK_PREFIX}state.hung|token|txid")

    assert "state.hung is blocking the event loop" in caplog_blocking.text
    assert "warned.wait(timeout=1)" in caplog_blocking.text


async def test_stop_as_the_loop_responds_still_reports_the_total(
    caplog_blocking: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    warned: threading.Event,
):
    """Report the total when a stop lands as the loop responds.

    The loop responds and stops the monitor between a timed-out responsiveness
    wait and the monitor's check for a stop.
    """
    monitor = EventLoopBlockingMonitor(
        asyncio.get_running_loop(), threshold=0.04, event_handlers={}
    )
    overtaken = threading.Event()

    class _OvertakenEvent(threading.Event):
        """An event the loop sets, and then stops the monitor, as a wait on it times out."""

        def wait(self, timeout: float | None = None) -> bool:
            """Wait, letting the loop and a stop overtake the first timeout after the warning.

            Args:
                timeout: Seconds to wait for the event.

            Returns:
                Whether the event was set before the wait timed out.
            """
            if super().wait(timeout):
                return True
            if warned.is_set() and not overtaken.is_set():
                self.set()
                monitor._stop.set()
                overtaken.set()
            return False

    # Only the responsiveness probes are created after this point.
    monkeypatch.setattr(
        blocking,
        "threading",
        SimpleNamespace(Event=_OvertakenEvent, Thread=threading.Thread),
    )

    async def hung_handler():  # noqa: RUF029
        """Hold the loop until the monitor's wait has been overtaken."""
        assert warned.wait(timeout=1)
        assert overtaken.wait(timeout=1)

    monitor.start()
    try:
        await asyncio.create_task(
            hung_handler(), name=f"{EVENT_TASK_PREFIX}state.hung|token|txid"
        )
    finally:
        monitor.stop()

    assert "Event handler state.hung blocked the event loop for" in caplog_blocking.text


async def test_stack_without_task_entry_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    """Do not describe a frame snapshot taken after the task left the loop."""
    monitor = EventLoopBlockingMonitor(
        asyncio.get_running_loop(), threshold=0.04, event_handlers={}
    )
    unrelated = sys._getframe()
    monkeypatch.setattr(
        sys,
        "_current_frames",
        lambda: {monitor._loop_thread_id: unrelated},
    )

    async def handler():  # noqa: RUF029
        """Describe the blocking from inside the event task.

        Returns:
            The description of the blocking.
        """
        return monitor._describe_blocking()

    description = await asyncio.create_task(
        handler(), name=f"{EVENT_TASK_PREFIX}state.moved|token|txid"
    )

    assert description is None


def _line_of(code: str) -> int:
    """Find the line of this file that holds exactly the given code.

    Args:
        code: The stripped text of the line.

    Returns:
        The 1-based line number of the first such line.
    """
    return next(
        number
        for number, line in enumerate(Path(__file__).read_text().splitlines(), 1)
        if line.strip() == code
    )


_SLEEP_FRAME = (
    __file__,
    _line_of("time.sleep(duration)  # noqa: ASYNC251"),
    "blocking_handler",
    "time.sleep(duration)  # noqa: ASYNC251",
)

_HINT = (
    "Move blocking I/O to a thread pool or use a non-blocking async API.\n"
    "Blocking stack (most recent call last):\n"
)


def _expected_warning(
    subject: str, location: str, note: str, frames: list[tuple[str, int, str, str]]
) -> str:
    """Build the complete warning the monitor logs when it detects blocking.

    Args:
        subject: The event handler or task the warning names.
        location: The app location of the blocking, or an empty string.
        note: The text after the scope note, or an empty string.
        frames: The expected frames as filename, line number, name and source line.

    Returns:
        The full text of the warning.
    """
    stack = traceback.format_list([
        traceback.FrameSummary(filename, lineno, name, line=line)
        for filename, lineno, name, line in frames
    ])
    return (
        f"{subject} is blocking the event loop{location} (threshold: 0.040s), "
        f"including state updates such as computed vars{note}. {_HINT}"
        f"{''.join(stack).rstrip()}"
    )


def _expected_total(subject: str) -> re.Pattern[str]:
    """Build the pattern of the line that reports the total blocking time.

    Args:
        subject: The event handler or task the line names.

    Returns:
        The pattern, with the duration left open.
    """
    return re.compile(
        rf"{re.escape(subject)} blocked the event loop for \d+\.\d{{3}}s in total\."
    )


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the full text of every captured log record.

    Args:
        caplog: The log capture fixture.

    Returns:
        The formatted messages, in order.
    """
    return [record.getMessage() for record in caplog.records]


def _library_handler(
    probe_queued: threading.Event, module: str = "thirdparty_pkg.handlers"
) -> tuple[Callable[[], Coroutine[Any, Any, None]], RegisteredEventHandler]:
    """Create a blocking handler that lives outside the app, in an installed package.

    Args:
        probe_queued: Set once the monitor queues a responsiveness probe.
        module: The module the handler claims to be defined in.

    Returns:
        The handler function and its registration.
    """
    handler = _blocking(probe_queued, _BLOCK_SECONDS)
    handler.__module__ = module
    return handler, RegisteredEventHandler(handler=EventHandler(fn=handler), states=())


_LIBRARY_SUBJECT = (
    "Event handler _blocking.<locals>.blocking_handler (from package thirdparty_pkg)"
)
_LIBRARY_NOTE = ", with no app code in the stack (reported once per process)"


async def test_library_handler_names_its_package(
    caplog_blocking: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Name the package of a handler with no app code in its blocking stack."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)
    handler, registered = _library_handler(probe_queued)

    await _run_monitored(
        handler,
        f"{EVENT_TASK_PREFIX}state.login|token|txid",
        {"state.login": registered},
        app_root=f"{tmp_path}{os.sep}",
    )

    warning, total = _messages(caplog_blocking)
    assert warning == _expected_warning(
        _LIBRARY_SUBJECT, "", _LIBRARY_NOTE, [_SLEEP_FRAME]
    )
    assert _expected_total(_LIBRARY_SUBJECT).fullmatch(total)


async def test_library_handler_stack_starts_at_the_handler(
    caplog_blocking: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Leave the frames that call a library handler out of the stack."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)
    handler, registered = _library_handler(probe_queued)

    async def framework_wrapper():
        """Call the handler, as the framework does."""
        await handler()

    await _run_monitored(
        framework_wrapper,
        f"{EVENT_TASK_PREFIX}state.login|token|txid",
        {"state.login": registered},
        app_root=f"{tmp_path}{os.sep}",
    )

    warning, total = _messages(caplog_blocking)
    assert warning == _expected_warning(
        _LIBRARY_SUBJECT, "", _LIBRARY_NOTE, [_SLEEP_FRAME]
    )
    assert _expected_total(_LIBRARY_SUBJECT).fullmatch(total)


async def test_library_handler_is_reported_once_per_process(
    caplog_blocking: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Report a blocking handler from a package only the first time."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)
    handler, registered = _library_handler(probe_queued)

    for _ in range(3):
        await _run_monitored(
            handler,
            f"{EVENT_TASK_PREFIX}state.login|token|txid",
            {"state.login": registered},
            app_root=f"{tmp_path}{os.sep}",
        )

    warning, total = _messages(caplog_blocking)
    assert warning == _expected_warning(
        _LIBRARY_SUBJECT, "", _LIBRARY_NOTE, [_SLEEP_FRAME]
    )
    assert _expected_total(_LIBRARY_SUBJECT).fullmatch(total)
    assert len(caplog_blocking.records) == 2


async def test_app_handler_is_reported_every_time(
    caplog_blocking: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """Keep reporting a blocking handler defined in the app."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)
    handler = _blocking(probe_queued, _BLOCK_SECONDS)
    registered = RegisteredEventHandler(handler=EventHandler(fn=handler), states=())

    for _ in range(2):
        await _run_monitored(
            handler,
            f"{EVENT_TASK_PREFIX}state.blocking|token|txid",
            {"state.blocking": registered},
        )

    subject = "Event handler _blocking.<locals>.blocking_handler"
    location = (
        f" at {__file__[len(f'{Path.cwd()}{os.sep}') :]}:{_SLEEP_FRAME[1]}"
        " in blocking_handler"
    )
    expected = _expected_warning(subject, location, "", [_SLEEP_FRAME])
    first, first_total, second, second_total = _messages(caplog_blocking)
    assert first == second == expected
    assert _expected_total(subject).fullmatch(first_total)
    assert _expected_total(subject).fullmatch(second_total)


async def test_unregistered_event_outside_the_app_keeps_the_full_report(
    caplog_blocking: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Name no package when no handler is registered for the event."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)

    await _run_monitored(
        _blocking(probe_queued, _BLOCK_SECONDS),
        f"{EVENT_TASK_PREFIX}state.unknown|token|txid",
        app_root=f"{tmp_path}{os.sep}",
    )

    warning, total = _messages(caplog_blocking)
    assert warning == _expected_warning(
        "Event handler state.unknown", "", "", [_SLEEP_FRAME]
    )
    assert _expected_total("Event handler state.unknown").fullmatch(total)


async def test_library_handler_missing_from_the_stack_keeps_the_full_stack(
    caplog_blocking: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Keep every frame when the registered handler is not among them."""
    probe_queued = _track_queued_probe(asyncio.get_running_loop(), monkeypatch)
    blocking_fn = _blocking(probe_queued, _BLOCK_SECONDS)

    async def registered_handler():
        """A handler that is registered but not the one that blocks."""

    registered_handler.__module__ = "thirdparty_pkg.handlers"
    registered = RegisteredEventHandler(
        handler=EventHandler(fn=registered_handler), states=()
    )

    async def framework_wrapper():
        """Block through a function other than the registered handler."""
        await blocking_fn()

    await _run_monitored(
        framework_wrapper,
        f"{EVENT_TASK_PREFIX}state.login|token|txid",
        {"state.login": registered},
        app_root=f"{tmp_path}{os.sep}",
    )

    subject = (
        "Event handler test_library_handler_missing_from_the_stack_keeps_the_full_stack"
        ".<locals>.registered_handler (from package thirdparty_pkg)"
    )
    warning, total = _messages(caplog_blocking)
    assert warning == _expected_warning(
        subject,
        "",
        _LIBRARY_NOTE,
        [
            (
                __file__,
                _line_of("await blocking_fn()"),
                "framework_wrapper",
                "await blocking_fn()",
            ),
            _SLEEP_FRAME,
        ],
    )
    assert _expected_total(subject).fullmatch(total)
