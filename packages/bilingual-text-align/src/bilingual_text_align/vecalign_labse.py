"""Adapter for a persistent isolated Vecalign + LaBSE worker."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .containment import (
    ContainmentConfig,
    ContainmentMatch,
    locate_from_similarity,
)
from .model import AlignmentLink, TextUnit
from .resource_paths import default_model_storage
from .semantic_runtime import require_semantic_runtime
from .worker_transport import DEFAULT_WORKER_STARTUP_TIMEOUT as DEFAULT_WORKER_STARTUP_TIMEOUT
from .worker_transport import JsonLinesWorker, WorkerTransport

DEFAULT_MODEL = "sentence-transformers/LaBSE"
DEFAULT_MODEL_REVISION = "836121a0533e5664b21c7aacc5d22951f2b8b25b"


class VecalignLabseAligner:
    def __init__(
        self,
        python: str | Path,
        *,
        model: str = DEFAULT_MODEL,
        model_revision: str = DEFAULT_MODEL_REVISION,
        model_storage: str | Path | None = None,
        cache_dir: str | Path | None = None,
        alignment_max_size: int = 8,
        batch_size: int = 16,
        startup_timeout: float = DEFAULT_WORKER_STARTUP_TIMEOUT,
        transport: WorkerTransport | None = None,
        worker_command: Sequence[str] | None = None,
        runtime_checker: Callable[[str | Path, str | Path], None] = require_semantic_runtime,
    ) -> None:
        if alignment_max_size < 2 or batch_size < 1:
            raise ValueError("alignment_max_size must be >= 2 and batch_size must be positive")
        self._metadata: dict[str, object] = {
            "name": "vecalign-labse",
            "version": 1,
            "model": model,
            "model_revision": model_revision,
            "alignment_max_size": alignment_max_size,
            "embedding_batch_size": batch_size,
            "score_kind": "exp(-vecalign_cost); diagnostic, not calibrated confidence",
        }
        storage = default_model_storage() if model_storage is None else Path(model_storage)
        command = [
            *(
                worker_command
                if worker_command is not None
                else (str(Path(python).expanduser()), str(Path(__file__).with_name("worker.py")))
            ),
            "--model",
            model,
            "--model-revision",
            model_revision,
            "--alignment-max-size",
            str(alignment_max_size),
            "--batch-size",
            str(batch_size),
        ]
        command.extend(["--model-storage", str(storage.expanduser())])
        if cache_dir is not None:
            command.extend(["--embedding-cache", str(Path(cache_dir).expanduser())])
        if transport is None:
            if worker_command is None:
                runtime_checker(python, storage)
            elif not Path(worker_command[0]).is_file():
                raise RuntimeError(f"bundled alignment worker not found at {worker_command[0]}")
            transport = JsonLinesWorker(command, startup_timeout=startup_timeout)
        self._transport = transport

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    def align(
        self, first: Sequence[TextUnit], second: Sequence[TextUnit]
    ) -> Sequence[AlignmentLink]:
        rows = self._transport.request(
            [item.text for item in first], [item.text for item in second]
        )
        links = []
        for row in rows:
            left = tuple(_integer_list(row, "first_indices"))
            right = tuple(_integer_list(row, "second_indices"))
            cost = max(_number(row, "cost"), 0.0)
            links.append(AlignmentLink(left, right, round(math.exp(-cost), 4)))
        return links

    def locate(
        self,
        query_texts: Sequence[str],
        corpus_texts: Sequence[str],
        config: ContainmentConfig | None = None,
    ) -> ContainmentMatch:
        """Locate query anchors inside corpus anchors with the shared model."""

        if not query_texts or not corpus_texts:
            raise ValueError("containment anchor sequences must be non-empty")
        similarities = self.similarities(query_texts, corpus_texts)
        return locate_from_similarity(similarities, config)

    def similarities(
        self, query_texts: Sequence[str], corpus_texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        """Return semantic similarities from the persistent worker."""

        if not query_texts or not corpus_texts:
            raise ValueError("similarity text sequences must be non-empty")
        similarities = self._transport.similarities(query_texts, corpus_texts)
        if len(similarities) != len(query_texts) or any(
            len(row) != len(corpus_texts) for row in similarities
        ):
            raise RuntimeError("alignment worker returned invalid similarities")
        return similarities

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> VecalignLabseAligner:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _integer_list(row: Mapping[str, object], key: str) -> list[int]:
    value = row.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise RuntimeError(f"worker field {key} must be a list")
    return value


def _number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"worker field {key} must be numeric")
    return float(value)
