"""Public EPUB bilingual-footnote API."""

from .model import Artifact, Book, ScopeSelection
from .pipeline import build_epub
from .protocol import BookReader, EpubRenderer
from .service import EpubFootnoteService
from .verification import VerificationReport, verify_epub

__all__ = [
    "Artifact",
    "Book",
    "BookReader",
    "EpubFootnoteService",
    "EpubRenderer",
    "ScopeSelection",
    "VerificationReport",
    "build_epub",
    "verify_epub",
]
