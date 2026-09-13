"""Semantic embedding, caching, and Vecalign computation."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from math import ceil
from pathlib import Path
from random import seed
from typing import Any

from .embedding_cache import EmbeddingCache

MODEL_STORAGE_MARKER = ".bilingual-footnotes-model"


def _cache_key(model: str, revision: str, maximum: int, sentences: list[str]) -> str:
    value = json.dumps(
        {"model": model, "revision": revision, "maximum": maximum, "sentences": sentences},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _text_cache_key(model: str, revision: str, texts: list[str]) -> str:
    value = json.dumps(
        {"model": model, "revision": revision, "kind": "texts", "texts": texts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


class Worker:
    def __init__(
        self,
        model: str,
        revision: str,
        maximum: int,
        batch_size: int,
        cache: EmbeddingCache,
        model_storage: Path | None = None,
        module_loader: Callable[[str], Any] = importlib.import_module,
        *,
        local_files_only: bool = True,
    ) -> None:
        self.np = module_loader("numpy")
        transformers = module_loader("sentence_transformers")
        self.dp = module_loader("vecalign.dp_utils")
        self.model_name, self.model_revision, self.maximum, self.batch_size, self.cache = (
            model,
            revision,
            maximum,
            batch_size,
            cache,
        )
        options: dict[str, Any] = {
            "revision": revision,
            "device": "cpu",
            "trust_remote_code": False,
            "local_files_only": local_files_only,
            "model_kwargs": {"use_safetensors": True},
        }
        if model_storage is not None:
            model_storage.mkdir(parents=True, exist_ok=True)
            options["cache_folder"] = str(model_storage)
        self.model = transformers.SentenceTransformer(model, **options)
        if model_storage is not None:
            (model_storage / MODEL_STORAGE_MARKER).touch(exist_ok=True)

    def embeddings(self, sentences: list[str]) -> Any:
        key = _cache_key(self.model_name, self.model_revision, self.maximum, sentences)
        cached = self.cache.load(key, (self.maximum, len(sentences)))
        if cached is not None:
            return cached
        overlaps = list(self.dp.yield_overlaps(sentences, self.maximum))
        encoded = self.model.encode(
            overlaps, batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        encoded = self.np.asarray(encoded, dtype=self.np.float32).reshape(
            self.maximum, len(sentences), -1
        )
        self.cache.store(key, encoded)
        return encoded

    def text_embeddings(self, texts: list[str]) -> Any:
        key = _text_cache_key(self.model_name, self.model_revision, texts)
        cached = self.cache.load(key, (len(texts),))
        if cached is not None:
            return cached
        encoded = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        encoded = self.np.asarray(encoded, dtype=self.np.float32)
        self.cache.store(key, encoded)
        return encoded

    def similarities(self, query: list[str], corpus: list[str]) -> list[list[float]]:
        values = self.text_embeddings(query) @ self.text_embeddings(corpus).T
        return [[float(value) for value in row] for row in values.tolist()]

    def align(self, first: list[str], second: list[str]) -> list[dict[str, Any]]:
        seed(42)
        self.np.random.seed(42)
        stack = self.dp.vecalign(
            vecs0=self.embeddings(first),
            vecs1=self.embeddings(second),
            final_alignment_types=self.dp.make_alignment_types(self.maximum),
            del_percentile_frac=0.2,
            width_over2=ceil(self.maximum / 2) + 5,
            max_size_full_dp=300,
            costs_sample_size=20000,
            num_samps_for_norm=100,
        )
        final = stack[0]
        return [
            {"first_indices": list(left), "second_indices": list(right), "cost": float(cost)}
            for (left, right), cost in zip(
                final["final_alignments"], final["alignment_scores"], strict=True
            )
        ]
