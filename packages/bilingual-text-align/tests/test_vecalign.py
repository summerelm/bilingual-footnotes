from __future__ import annotations

import io
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, cast

import pytest

from bilingual_text_align.model import TextUnit
from bilingual_text_align.resource_paths import MODEL_STORAGE_MARKER
from bilingual_text_align.vecalign_labse import VecalignLabseAligner
from bilingual_text_align.worker_transport import JsonLinesWorker


class FakeTransport:
    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        similarities: Sequence[Sequence[float]] = (),
    ) -> None:
        self.rows = rows
        self.similarity_rows = similarities
        self.requests: list[tuple[Sequence[str], Sequence[str]]] = []
        self.similarity_requests: list[tuple[Sequence[str], Sequence[str]]] = []
        self.closed = False

    def request(
        self, first: Sequence[str], second: Sequence[str]
    ) -> Sequence[Mapping[str, object]]:
        self.requests.append((first, second))
        return self.rows

    def similarities(
        self, query: Sequence[str], corpus: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        self.similarity_requests.append((query, corpus))
        return self.similarity_rows

    def close(self) -> None:
        self.closed = True


def test_adapter_maps_worker_indices_and_closes() -> None:
    transport = FakeTransport([{"first_indices": [0, 1], "second_indices": [0], "cost": 0.25}])
    with VecalignLabseAligner("python", transport=transport) as aligner:
        links = aligner.align(
            [TextUnit(0, "Un", 0), TextUnit(1, "Deux", 0)], [TextUnit(0, "One two", 0)]
        )
    assert links[0].first_indices == (0, 1)
    assert links[0].score == 0.7788
    assert transport.requests == [(["Un", "Deux"], ["One two"])]
    assert transport.closed
    assert aligner.metadata["name"] == "vecalign-labse"
    assert aligner.metadata["model_revision"] == "836121a0533e5664b21c7aacc5d22951f2b8b25b"
    assert aligner.metadata["embedding_batch_size"] == 16


def test_adapter_validates_options_and_worker_rows() -> None:
    with pytest.raises(ValueError):
        VecalignLabseAligner("python", alignment_max_size=1)
    aligner = VecalignLabseAligner("python", transport=FakeTransport([{"first_indices": "bad"}]))
    with pytest.raises(RuntimeError, match="must be a list"):
        aligner.align([TextUnit(0, "x", 0)], [TextUnit(0, "y", 0)])


def test_adapter_fails_before_transport_when_model_is_missing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    worker_python = tmp_path / "worker-python"

    with pytest.raises(RuntimeError, match="semantic model is not installed.*align-setup"):
        VecalignLabseAligner(worker_python, model_storage=model)

    model.mkdir()
    (model / MODEL_STORAGE_MARKER).touch()
    (model / "weights.safetensors").write_bytes(b"model")
    with pytest.raises(RuntimeError, match="worker Python not found.*align-setup"):
        VecalignLabseAligner(worker_python, model_storage=model)


def test_adapter_locates_query_inside_corpus() -> None:
    transport = FakeTransport(
        [],
        [
            [0.0, 0.95, 0.0, 0.0],
            [0.0, 0.0, 0.90, 0.0],
        ],
    )
    aligner = VecalignLabseAligner("python", transport=transport)

    match = aligner.locate(["un", "deux"], ["x", "one", "two", "y"])

    assert match.accepted
    assert (match.corpus_start, match.corpus_end) == (1, 3)
    assert transport.similarity_requests == [(["un", "deux"], ["x", "one", "two", "y"])]


def test_adapter_exposes_validated_similarities() -> None:
    transport = FakeTransport([], [[0.9, 0.1]])
    aligner = VecalignLabseAligner("python", transport=transport)

    assert aligner.similarities(["query"], ["match", "other"]) == [[0.9, 0.1]]
    with pytest.raises(ValueError, match="non-empty"):
        aligner.similarities([], ["match"])


def test_adapter_rejects_invalid_similarity_shape() -> None:
    aligner = VecalignLabseAligner("python", transport=FakeTransport([], [[1.0]]))
    with pytest.raises(RuntimeError, match="invalid similarities"):
        aligner.locate(["one", "two"], ["un", "deux"])
    with pytest.raises(ValueError, match="non-empty"):
        aligner.locate([], ["un"])


class FakeInput(io.StringIO):
    def close(self) -> None:
        self.flush()


class FakeProcess:
    def __init__(self, output: str, *, timeout: bool = False) -> None:
        self.stdin: IO[str] | None = FakeInput()
        self.stdout: IO[str] | None = io.StringIO(output)
        self.stderr: IO[str] | None = io.StringIO("")
        self.timeout = timeout
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if self.timeout and not self.terminated:
            raise subprocess.TimeoutExpired("worker", timeout or 0)
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class FakeFactory:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.commands: list[Sequence[str]] = []

    def __call__(self, command: Sequence[str]) -> FakeProcess:
        self.commands.append(command)
        return self.process


def test_json_lines_transport_reuses_worker_and_closes() -> None:
    process = FakeProcess(
        '{"status":"ready"}\n'
        '{"alignments":[{"first_indices":[0],"second_indices":[0],"cost":0.1}]}\n',
        timeout=True,
    )
    factory = FakeFactory(process)
    worker = JsonLinesWorker(["python", "worker.py"], factory)
    assert worker.request(["un"], ["one"])[0]["cost"] == 0.1
    worker.close()
    worker.close()
    assert len(factory.commands) == 1
    assert isinstance(process.stdin, io.StringIO)
    assert '"command":"close"' in process.stdin.getvalue()
    assert process.terminated


def test_json_lines_transport_requests_similarities() -> None:
    process = FakeProcess('{"status":"ready"}\n{"similarities":[[0.9,0.1]]}\n')
    worker = JsonLinesWorker(["python", "worker.py"], FakeFactory(process))
    assert worker.similarities(["query"], ["match", "other"]) == [[0.9, 0.1]]
    assert isinstance(process.stdin, io.StringIO)
    assert '"command": "similarities"' in process.stdin.getvalue()


def test_json_lines_transport_rejects_worker_failures() -> None:
    error = JsonLinesWorker(
        ["worker"], FakeFactory(FakeProcess('{"status":"ready"}\n{"error":"bad"}\n'))
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        error.request(["x"], ["y"])
    invalid = JsonLinesWorker(
        ["worker"], FakeFactory(FakeProcess('{"status":"ready"}\n{"alignments":{}}\n'))
    )
    with pytest.raises(RuntimeError, match="invalid alignments"):
        invalid.request(["x"], ["y"])
    invalid_similarities = JsonLinesWorker(
        ["worker"], FakeFactory(FakeProcess('{"status":"ready"}\n{"similarities":{}}\n'))
    )
    with pytest.raises(RuntimeError, match="invalid similarities"):
        invalid_similarities.similarities(["x"], ["y"])
    missing = JsonLinesWorker(["worker"], FakeFactory(FakeProcess("")))
    with pytest.raises(RuntimeError, match="without a response"):
        missing.request(["x"], ["y"])

    failed = FakeProcess("")
    failed.stderr = io.StringIO("python: can't open file 'missing-worker.py'\n")
    detailed = JsonLinesWorker(["worker"], FakeFactory(failed))
    with pytest.raises(RuntimeError, match="missing-worker.py"):
        detailed.request(["x"], ["y"])

    startup = JsonLinesWorker(["worker"], FakeFactory(FakeProcess('{"status":"wrong"}\n')))
    with pytest.raises(RuntimeError, match="startup"):
        startup.request(["x"], ["y"])

    non_object = JsonLinesWorker(["worker"], FakeFactory(FakeProcess("[]\n")))
    with pytest.raises(RuntimeError, match="non-object"):
        non_object.request(["x"], ["y"])


def test_json_lines_transport_times_out_during_startup() -> None:
    release = threading.Event()

    class BlockingOutput:
        def readline(self, size: int = -1) -> str:
            del size
            release.wait()
            return ""

    process = FakeProcess("", timeout=True)
    process.stdout = cast(IO[str], BlockingOutput())
    worker = JsonLinesWorker(["worker"], FakeFactory(process), startup_timeout=0.01)

    with pytest.raises(RuntimeError, match="did not start within.*worker Python path"):
        worker.request(["x"], ["y"])

    assert process.terminated
    release.set()


def test_json_lines_transport_requires_positive_startup_timeout() -> None:
    with pytest.raises(ValueError, match="startup timeout must be positive"):
        JsonLinesWorker(["worker"], startup_timeout=0)


def test_json_lines_transport_explains_missing_worker_python() -> None:
    class MissingFactory:
        def __call__(self, command: Sequence[str]) -> FakeProcess:
            del command
            raise FileNotFoundError("missing")

    worker = JsonLinesWorker(["missing-python", "worker.py"], MissingFactory())

    with pytest.raises(RuntimeError, match="bilingual-align-setup.*--aligner-python"):
        worker.request(["x"], ["y"])


def test_json_lines_transport_runs_a_real_process() -> None:
    script = (
        "import json,sys; print(json.dumps({'status':'ready'}),flush=True); "
        "sys.stdin.readline(); print(json.dumps({'alignments':[]}),flush=True); "
        "sys.stdin.readline()"
    )
    worker = JsonLinesWorker([sys.executable, "-c", script])
    assert worker.request(["x"], ["y"]) == []
    worker.close()


def test_json_lines_transport_handles_missing_stdin_and_broken_pipe() -> None:
    missing_stdin = FakeProcess('{"status":"ready"}\n')
    missing_stdin.stdin = None
    worker = JsonLinesWorker(["worker"], FakeFactory(missing_stdin))
    with pytest.raises(RuntimeError, match="stdin"):
        worker.request(["x"], ["y"])

    class BrokenInput(FakeInput):
        def write(self, value: str) -> int:
            del value
            raise BrokenPipeError

    broken = FakeProcess("")
    broken.stdin = BrokenInput()
    closer = JsonLinesWorker(["worker"], FakeFactory(broken))
    closer._process = broken
    closer.close()


def test_json_lines_transport_kills_worker_that_ignores_termination() -> None:
    class StubbornProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            if not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout or 0)
            return 0

    process = StubbornProcess("")
    worker = JsonLinesWorker(["worker"], FakeFactory(process))
    worker._process = process

    worker.close()

    assert process.terminated
    assert process.killed


def test_json_lines_transport_reports_worker_that_cannot_be_killed() -> None:
    class UnstoppableProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("worker", timeout or 0)

    process = UnstoppableProcess("")
    worker = JsonLinesWorker(["worker"], FakeFactory(process))
    worker._process = process

    with pytest.raises(RuntimeError, match="could not be stopped"):
        worker.close()

    assert process.terminated
    assert process.killed


def test_adapter_rejects_invalid_cost() -> None:
    aligner = VecalignLabseAligner(
        "python",
        transport=FakeTransport([{"first_indices": [0], "second_indices": [0], "cost": True}]),
    )
    with pytest.raises(RuntimeError, match="numeric"):
        aligner.align([TextUnit(0, "x", 0)], [TextUnit(0, "y", 0)])
