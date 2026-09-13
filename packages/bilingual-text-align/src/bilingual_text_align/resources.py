"""Public resource-management API."""

from .model_management import prepare_model, prepare_model_command, semantic_model_installed
from .resource_paths import (
    EMBEDDING_CACHE_MARKER,
    MODEL_STORAGE_MARKER,
    default_embedding_cache,
    default_model_storage,
    default_storage_root,
    default_worker_python,
)
from .resource_storage import (
    StoredResource,
    clear_managed_resource,
    format_size,
    inspect_resource,
    mark_resource,
    prune_embedding_cache,
    resource_activity_marker,
)

__all__ = [
    "EMBEDDING_CACHE_MARKER",
    "MODEL_STORAGE_MARKER",
    "StoredResource",
    "clear_managed_resource",
    "default_embedding_cache",
    "default_model_storage",
    "default_storage_root",
    "default_worker_python",
    "format_size",
    "inspect_resource",
    "mark_resource",
    "prepare_model",
    "prepare_model_command",
    "prune_embedding_cache",
    "resource_activity_marker",
    "semantic_model_installed",
]
