"""Test configurable index backend factories and the default SQLite backend."""

import sys
from types import ModuleType

import pytest

from conftest import FakeEmbeddingProvider
from isthmus.backends import open_index
from isthmus.index import BackendIndex, SQLiteIndex
from isthmus.models import Chunk, Document
from isthmus.pipeline import Isthmus


class CustomBackend(BackendIndex):
    def __init__(self, provider):
        """Initialize the custom backend fixture and its closed-state flag."""
        super().__init__(provider)
        self.provider_id = "custom"
        self.closed = False

    def add_sector(self, sector_id):
        """Add a sector to the custom backend fixture."""
        self.sectors.add(sector_id)

    def remove_sector(self, sector_id):
        """Remove a sector from the custom backend fixture."""
        self.sectors.remove(sector_id)

    def remove_document(self, document_id):
        """Remove a document from the custom backend fixture."""
        del self.documents[document_id]

    def close(self):
        """Mark the custom backend fixture as closed."""
        self.closed = True


def test_custom_backend_factory_supports_retrieval(monkeypatch):
    """Verify that custom backend factory supports retrieval."""
    module = ModuleType("test_custom_backend")
    seen = []

    def create(provider, database):
        """Record the database path and return a custom backend."""
        seen.append(database)
        return CustomBackend(provider)

    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("ISTHMUS_INDEX_FACTORY", "test_custom_backend:create")
    with open_index(FakeEmbeddingProvider(), "my-index") as index:
        index.add_document(Document("doc", "Policy", "sector"), [
            Chunk("chunk", "doc", "sector", "travel policy", 1)
        ])
        assert Isthmus(index).query("travel").context[0].id == "chunk"
    assert index.closed
    assert seen == ["my-index"]


def test_invalid_backend_factory_fails_explicitly(monkeypatch):
    """Verify that invalid backend factory fails explicitly."""
    monkeypatch.setenv("ISTHMUS_INDEX_FACTORY", "invalid")
    with pytest.raises(ValueError, match="module:function"):
        open_index(FakeEmbeddingProvider(), "unused")


def test_default_backend_remains_sqlite(monkeypatch, tmp_path):
    """Verify that default backend remains sqlite."""
    monkeypatch.delenv("ISTHMUS_INDEX_FACTORY", raising=False)
    with open_index(FakeEmbeddingProvider(), tmp_path / "test.db") as index:
        assert isinstance(index, SQLiteIndex)
