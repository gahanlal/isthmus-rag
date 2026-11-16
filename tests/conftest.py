"""Define shared fixtures and test doubles for the Isthmus test suite."""

class FakeEmbeddingProvider:
    provider_id = "test-provider"

    def embed(self, text):
        """Return a deterministic test embedding for the supplied text."""
        return (1.0, 0.0)
