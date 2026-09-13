"""Persistent JSON-lines transport for the isolated alignment worker."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from typing import IO, Protocol

DEFAULT_WORKER_STARTUP_TIMEOUT = 600.0


class WorkerTransport(Protocol):
    def request(
        self, first: Sequence[str], second: Sequence[str]
    ) -> Sequence[Mapping[str, object]]: ...

    def similarities(
        self, query: Sequence[str], corpus: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...

    def close(self) -> None: ...


class WorkerProcess(Protocol):
    stdin: IO[str] | None
    stdout: IO[str] | None
    stderr: IO[str] | None

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessFactory(Protocol):
    def __call__(self, command: Sequence[str]) -> WorkerProcess: ...


def _worker_creation_flags(platform: str = sys.platform) -> int:
    """Keep the bundled Windows worker from opening a second console window."""

    return 0x08000000 if platform == "win32" else 0


def _open_process(command: Sequence[str]) -> WorkerProcess:
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        creationflags=_worker_creation_flags(),
    )


class JsonLinesWorker:
    def __init__(
        self,
        command: Sequence[str],
        process_factory: ProcessFactory = _open_process,
        *,
        startup_timeout: float = DEFAULT_WORKER_STARTUP_TIMEOUT,
    ) -> None:
        if startup_timeout <= 0:
            raise ValueError("worker startup timeout must be positive")
        self._command = tuple(command)
        self._process_factory = process_factory
        self._startup_timeout = startup_timeout
        self._process: WorkerProcess | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_closed = threading.Event()

    def _start(self) -> WorkerProcess:
        if self._process is None:
            try:
                self._process = self._process_factory(self._command)
            except OSError as exc:
                raise RuntimeError(
                    f"could not start the alignment worker with {self._command[0]}; "
                    "run bilingual-align-setup and pass its Python path with --aligner-python"
                ) from exc
            self._capture_stderr(self._process.stderr)
            try:
                ready = self._read_with_timeout(self._process.stdout)
            except Exception:
                self.close()
                raise
            if ready.get("status") != "ready":
                self.close()
                raise RuntimeError(f"unexpected worker startup response: {ready}")
        return self._process

    def _read_with_timeout(self, stream: IO[str] | None) -> dict[str, object]:
        result: queue.Queue[dict[str, object] | BaseException] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                result.put(self._read(stream))
            except BaseException as exc:
                result.put(exc)

        threading.Thread(target=read, daemon=True).start()
        try:
            value = result.get(timeout=self._startup_timeout)
        except queue.Empty as exc:
            raise RuntimeError(
                f"alignment worker did not start within {self._startup_timeout:g} seconds; "
                "check the worker Python path, model download, and network access"
            ) from exc
        if isinstance(value, BaseException):
            raise value
        return value

    def _capture_stderr(self, stream: IO[str] | None) -> None:
        if stream is None:
            return

        def drain() -> None:
            try:
                for line in stream:
                    if detail := line.strip():
                        self._stderr_tail.append(detail)
            finally:
                self._stderr_closed.set()

        threading.Thread(target=drain, daemon=True).start()

    def _read(self, stream: IO[str] | None) -> dict[str, object]:
        if stream is None or not (line := stream.readline()):
            self._stderr_closed.wait(timeout=0.2)
            detail = self._stderr_tail[-1] if self._stderr_tail else None
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"alignment worker exited without a response{suffix}")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise RuntimeError("alignment worker returned a non-object response")
        return message

    def request(
        self, first: Sequence[str], second: Sequence[str]
    ) -> Sequence[Mapping[str, object]]:
        response = self._exchange({"first": first, "second": second})
        rows = response.get("alignments")
        if not isinstance(rows, list):
            raise RuntimeError("alignment worker returned invalid alignments")
        return rows

    def similarities(
        self, query: Sequence[str], corpus: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        response = self._exchange({"command": "similarities", "query": query, "corpus": corpus})
        rows = response.get("similarities")
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise RuntimeError("alignment worker returned invalid similarities")
        return rows

    def _exchange(self, message: Mapping[str, object]) -> dict[str, object]:
        process = self._start()
        if process.stdin is None:
            raise RuntimeError("alignment worker stdin is unavailable")
        process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response = self._read(process.stdout)
        if error := response.get("error"):
            raise RuntimeError(f"alignment worker failed: {error}")
        return response

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("alignment worker could not be stopped") from exc
