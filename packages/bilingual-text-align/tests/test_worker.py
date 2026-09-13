from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from bilingual_text_align import embedding_worker as engine_module
from bilingual_text_align import worker as protocol_module
from bilingual_text_align.embedding_cache import DisabledEmbeddingCache


class FakeArray:
    def __init__(self) -> None:
        self.reshape_calls: list[tuple[int, ...]] = []

    def reshape(self, *shape: int) -> FakeArray:
        self.reshape_calls.append(shape)
        return self

    @property
    def T(self) -> FakeArray:
        return self

    def __matmul__(self, other: FakeArray) -> FakeArray:
        assert other is self
        return self

    def tolist(self) -> list[list[float]]:
        return [[0.9]]


class FakeRandom:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def seed(self, value: int) -> None:
        self.seeds.append(value)


class FakeNumpy:
    float32 = "float32"

    def __init__(self) -> None:
        self.random = FakeRandom()
        self.array = FakeArray()

    def asarray(self, value: object, *, dtype: object) -> FakeArray:
        assert value == [[1.0], [2.0], [3.0], [4.0]]
        assert dtype == self.float32
        return self.array


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, bool, bool]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        self.calls.append((sentences, batch_size, show_progress_bar, normalize_embeddings))
        return [[1.0], [2.0], [3.0], [4.0]]


class FakeTransformers:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def SentenceTransformer(self, model: str, **options: object) -> FakeModel:
        self.calls.append((model, options))
        return self.model


class FakeDp:
    def __init__(self) -> None:
        self.vecalign_arguments: dict[str, object] = {}

    def yield_overlaps(self, sentences: list[str], maximum: int) -> list[str]:
        assert maximum == 2
        return [*sentences, " ".join(sentences), ""]

    def make_alignment_types(self, maximum: int) -> str:
        return f"types-{maximum}"

    def vecalign(self, **arguments: object) -> list[dict[str, object]]:
        self.vecalign_arguments = arguments
        return [
            {
                "final_alignments": [((0,), (0, 1)), ((1,), ())],
                "alignment_scores": [0.25, 0.75],
            }
        ]


class FakeCache:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.loads: list[tuple[str, tuple[int, ...]]] = []
        self.stores: list[tuple[str, object]] = []

    def load(self, key: str, expected_prefix: tuple[int, ...]) -> object | None:
        self.loads.append((key, expected_prefix))
        return self.values.get(key)

    def store(self, key: str, embeddings: object) -> None:
        self.stores.append((key, embeddings))
        self.values[key] = embeddings


def make_worker(
    cache: FakeCache | DisabledEmbeddingCache | None = None,
) -> tuple[engine_module.Worker, FakeNumpy, FakeModel, FakeTransformers, FakeDp, FakeCache]:
    numpy = FakeNumpy()
    model = FakeModel()
    transformers = FakeTransformers(model)
    dp = FakeDp()
    modules: dict[str, object] = {
        "numpy": numpy,
        "sentence_transformers": transformers,
        "vecalign.dp_utils": dp,
    }
    active_cache = FakeCache() if cache is None else cache
    instance = engine_module.Worker(
        "model", "revision", 2, 4, active_cache, module_loader=modules.__getitem__
    )
    assert isinstance(active_cache, FakeCache)
    return instance, numpy, model, transformers, dp, active_cache


def test_worker_uses_injected_cache_for_embeddings() -> None:
    instance, numpy, model, transformers, _dp, cache = make_worker()

    encoded = instance.embeddings(["Un.", "Deux."])
    cached = instance.embeddings(["Un.", "Deux."])

    assert transformers.calls == [
        (
            "model",
            {
                "revision": "revision",
                "device": "cpu",
                "trust_remote_code": False,
                "local_files_only": True,
                "model_kwargs": {"use_safetensors": True},
            },
        )
    ]
    assert model.calls == [(["Un.", "Deux.", "Un. Deux.", ""], 4, True, True)]
    assert encoded is numpy.array
    assert numpy.array.reshape_calls == [(2, 2, -1)]
    assert cache.stores == [(cache.loads[0][0], numpy.array)]
    assert cache.loads == [(cache.loads[0][0], (2, 2)), (cache.loads[0][0], (2, 2))]
    assert cached is numpy.array
    assert engine_module._cache_key("model", "revision", 2, ["é"]) != engine_module._cache_key(
        "model", "other-revision", 2, ["é"]
    )


def test_worker_uses_injected_cache_for_containment_embeddings() -> None:
    instance, numpy, model, _transformers, _dp, cache = make_worker()

    encoded = instance.text_embeddings(["anchor"])
    cached = instance.text_embeddings(["anchor"])

    assert encoded is numpy.array
    assert cached is numpy.array
    assert model.calls == [(["anchor"], 4, True, True)]
    assert cache.stores == [(cache.loads[0][0], numpy.array)]
    assert instance.similarities(["query"], ["corpus"]) == [[0.9]]
    assert engine_module._text_cache_key(
        "model", "revision", ["anchor"]
    ) != engine_module._text_cache_key("model", "other", ["anchor"])


def test_worker_uses_separate_model_storage(tmp_path: Path) -> None:
    numpy = FakeNumpy()
    model = FakeModel()
    transformers = FakeTransformers(model)
    modules: dict[str, object] = {
        "numpy": numpy,
        "sentence_transformers": transformers,
        "vecalign.dp_utils": FakeDp(),
    }
    engine_module.Worker(
        "model",
        "revision",
        2,
        4,
        DisabledEmbeddingCache(),
        tmp_path / "model",
        modules.__getitem__,
    )

    assert transformers.calls == [
        (
            "model",
            {
                "revision": "revision",
                "device": "cpu",
                "trust_remote_code": False,
                "local_files_only": True,
                "model_kwargs": {"use_safetensors": True},
                "cache_folder": str(tmp_path / "model"),
            },
        )
    ]
    assert (tmp_path / "model" / engine_module.MODEL_STORAGE_MARKER).is_file()


def test_worker_converts_vecalign_results_and_uses_fixed_parameters() -> None:
    instance, numpy, _model, _transformers, dp, _cache = make_worker()

    links = instance.align(["Un.", "Deux."], ["One.", "Two."])

    assert links == [
        {"first_indices": [0], "second_indices": [0, 1], "cost": 0.25},
        {"first_indices": [1], "second_indices": [], "cost": 0.75},
    ]
    assert numpy.random.seeds == [42]
    assert dp.vecalign_arguments["final_alignment_types"] == "types-2"
    assert dp.vecalign_arguments["width_over2"] == 6
    assert dp.vecalign_arguments["max_size_full_dp"] == 300


def test_worker_json_lines_protocol_reports_errors_and_closes() -> None:
    class ProtocolWorker:
        def __init__(self, *_args: object) -> None:
            pass

        def align(self, first: list[str], second: list[str]) -> list[dict[str, Any]]:
            if first == ["bad"]:
                raise ValueError("invalid request")
            return [{"first_indices": [0], "second_indices": [0], "cost": 0.1}]

        def similarities(self, query: list[str], corpus: list[str]) -> list[list[float]]:
            assert query == ["query"]
            assert corpus == ["corpus"]
            return [[0.9]]

    requests = "\n".join(
        (
            json.dumps({"first": ["un"], "second": ["one"]}),
            json.dumps({"command": "similarities", "query": ["query"], "corpus": ["corpus"]}),
            json.dumps({"command": "similarities", "query": [], "corpus": ["corpus"]}),
            json.dumps({"command": "similarities", "query": ["query"], "corpus": []}),
            json.dumps({"first": ["bad"], "second": ["request"]}),
            json.dumps({"command": "close"}),
        )
    )
    output = io.StringIO()
    protocol_module.main(
        [
            "--model",
            "model",
            "--model-revision",
            "revision",
            "--alignment-max-size",
            "4",
            "--batch-size",
            "8",
        ],
        input_stream=io.StringIO(requests + "\n"),
        output_stream=output,
        worker_factory=ProtocolWorker,
    )
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages == [
        {"status": "ready"},
        {"alignments": [{"first_indices": [0], "second_indices": [0], "cost": 0.1}]},
        {"similarities": [[0.9]]},
        {"error": "ValueError: containment query must be a non-empty list"},
        {"error": "ValueError: containment corpus must be a non-empty list"},
        {"error": "ValueError: invalid request"},
    ]


def test_prepare_model_allows_the_pinned_download() -> None:
    calls: list[tuple[object, ...]] = []

    class ProtocolWorker:
        def __init__(self, *arguments: object) -> None:
            calls.append(arguments)

        def align(self, _first: list[str], _second: list[str]) -> list[dict[str, Any]]:
            return []

        def similarities(self, _query: list[str], _corpus: list[str]) -> list[list[float]]:
            return []

    protocol_module.main(
        ["--model", "model", "--model-revision", "revision", "--prepare-model"],
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
        worker_factory=ProtocolWorker,
    )

    assert calls[0][-1] is False
