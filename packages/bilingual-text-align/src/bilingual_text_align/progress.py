"""User-visible elapsed-time heartbeat for long command-line work."""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import monotonic
from types import TracebackType


class ElapsedHeartbeat:
    """Periodically report the current label and elapsed time."""

    def __init__(
        self,
        label: Callable[[], str],
        interval: float = 60.0,
        write: Callable[[str], None] | None = None,
    ) -> None:
        if interval < 0:
            raise ValueError("progress interval must be zero or positive")
        self._label = label
        self._interval = interval
        self._write = write or _print_line
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> ElapsedHeartbeat:
        self._started = monotonic()
        if self._interval:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._done.wait(self._interval):
            elapsed = monotonic() - self._started
            self._write(f"[{self._label()}] Still working; {elapsed:.0f}s elapsed")


def _print_line(message: str) -> None:
    print(message, flush=True)
