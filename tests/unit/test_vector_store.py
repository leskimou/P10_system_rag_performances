from unittest.mock import MagicMock, patch

import faiss
import numpy as np

from utils.schemas import Chunk
from utils.vector_store import VectorStoreManager


def make_manager() -> VectorStoreManager:
    """Construit un VectorStoreManager sans appel réseau ni fichiers d'index réels."""
    with patch("utils.vector_store.Mistral", return_value=MagicMock()), patch(
        "utils.vector_store.MISTRAL_API_KEY", "fake-key"
    ), patch("utils.vector_store.os.path.exists", return_value=False):
        return VectorStoreManager()


def index_with_chunks(sample_chunk_metadata):
    """Petit index Faiss réel (2 vecteurs orthogonaux) pour tester search() sans API."""
    chunks = [
        Chunk(id="0_0", text="hello world", metadata=sample_chunk_metadata),
        Chunk(id="0_1", text="goodbye", metadata=sample_chunk_metadata),
    ]
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    return index, chunks


class TestSearch:
    def test_blank_query_returns_empty(self, sample_chunk_metadata):
        manager = make_manager()
        manager.index, manager.document_chunks = index_with_chunks(sample_chunk_metadata)

        assert manager.search("   ") == []

    def test_missing_index_returns_empty(self):
        manager = make_manager()
        manager.index = None
        manager.document_chunks = []

        assert manager.search("some query") == []

    def test_missing_api_key_returns_empty(self, sample_chunk_metadata):
        manager = make_manager()
        manager.index, manager.document_chunks = index_with_chunks(sample_chunk_metadata)

        with patch("utils.vector_store.MISTRAL_API_KEY", None):
            assert manager.search("some query") == []

    def test_returns_matching_chunk_sorted_by_score(self, sample_chunk_metadata):
        manager = make_manager()
        manager.index, manager.document_chunks = index_with_chunks(sample_chunk_metadata)

        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=[1.0, 0.0, 0.0])]
        manager.mistral_client.embeddings.create.return_value = fake_response

        results = manager.search("hello", k=2)

        assert len(results) == 2
        assert results[0].text == "hello world"
        assert results[0].score > results[1].score

    def test_k_limits_number_of_results(self, sample_chunk_metadata):
        manager = make_manager()
        manager.index, manager.document_chunks = index_with_chunks(sample_chunk_metadata)

        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=[1.0, 0.0, 0.0])]
        manager.mistral_client.embeddings.create.return_value = fake_response

        results = manager.search("hello", k=1)

        assert len(results) == 1

    def test_min_score_filters_low_similarity(self, sample_chunk_metadata):
        manager = make_manager()
        manager.index, manager.document_chunks = index_with_chunks(sample_chunk_metadata)

        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=[1.0, 0.0, 0.0])]
        manager.mistral_client.embeddings.create.return_value = fake_response

        results = manager.search("hello", k=2, min_score=0.5)

        assert len(results) == 1
        assert results[0].text == "hello world"
        assert results[0].score >= 50
