"""JSON-lines Vecalign worker loaded only by the optional semantic environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Protocol

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bilingual_text_align.embedding_cache import DisabledEmbeddingCache
from bilingual_text_align.embedding_worker import Worker
from bilingual_text_align.npy_embedding_cache import NpyEmbeddingCache


class AlignmentWorker(Protocol):
    def align(self, first: list[str], second: list[str]) -> list[dict[str, Any]]: ...

    def similarities(self, query: list[str], corpus: list[str]) -> list[list[float]]: ...


def build_worker(
    model: str,
    revision: str,
    maximum: int,
    batch_size: int,
    cache_path: Path | None,
    model_storage: Path | None,
    local_files_only: bool,
) -> Worker:
    cache = (
        NpyEmbeddingCache(cache_path, importlib.import_module("numpy"))
        if cache_path is not None
        else DisabledEmbeddingCache()
    )
    return Worker(
        model,
        revision,
        maximum,
        batch_size,
        cache,
        model_storage,
        local_files_only=local_files_only,
    )


def main(
    argv: list[str] | None = None,
    *,
    input_stream: IO[str] = sys.stdin,
    output_stream: IO[str] = sys.stdout,
    worker_factory: Callable[..., AlignmentWorker] = build_worker,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--alignment-max-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--model-storage", type=Path)
    parser.add_argument("--prepare-model", action="store_true")
    args = parser.parse_args(argv)
    activity = {
        path / f".active-{os.getpid()}"
        for path in (args.embedding_cache, args.model_storage)
        if path is not None
    }
    for marker in activity:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
    try:
        worker = worker_factory(
            args.model,
            args.model_revision,
            args.alignment_max_size,
            args.batch_size,
            args.embedding_cache,
            args.model_storage,
            not args.prepare_model,
        )
        if args.prepare_model:
            return
        print(json.dumps({"status": "ready"}), file=output_stream, flush=True)
        for line in input_stream:
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    return
                if request.get("command") == "similarities":
                    query = request.get("query")
                    corpus = request.get("corpus")
                    if not isinstance(query, list) or not query:
                        raise ValueError("containment query must be a non-empty list")
                    if not isinstance(corpus, list) or not corpus:
                        raise ValueError("containment corpus must be a non-empty list")
                    print(
                        json.dumps(
                            {
                                "similarities": worker.similarities(
                                    [str(item) for item in query],
                                    [str(item) for item in corpus],
                                )
                            }
                        ),
                        file=output_stream,
                        flush=True,
                    )
                    continue
                print(
                    json.dumps({"alignments": worker.align(request["first"], request["second"])}),
                    file=output_stream,
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                    file=output_stream,
                    flush=True,
                )
    finally:
        for marker in activity:
            marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
