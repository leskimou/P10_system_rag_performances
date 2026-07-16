"""Tests d'intégration : appels réels à l'API Mistral.

Auto-skippés si MISTRAL_API_KEY n'est pas défini (via .env ou l'environnement).
Nécessitent un index Faiss déjà construit dans vector_db/ (make index) pour le
test de recherche vectorielle.
"""
import pytest

from utils.config import MISTRAL_API_KEY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MISTRAL_API_KEY, reason="MISTRAL_API_KEY non défini"),
]


def test_vector_store_search_returns_real_results():
    from utils.vector_store import VectorStoreManager

    manager = VectorStoreManager()
    if manager.index is None:
        pytest.skip("Aucun index Faiss trouvé dans vector_db/ (lancez `make index` d'abord)")

    results = manager.search("Quelles sont les tendances de la ligue ?", k=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    for result in results:
        assert result.text
        assert 0 <= result.score <= 100


def test_chatbot_generate_answer_calls_real_llm():
    from utils.chatbot import generate_answer

    answer, tool_contexts = generate_answer("Bonjour, qui es-tu ?", search_results=[])

    assert isinstance(answer, str)
    assert answer.strip() != ""
    assert isinstance(tool_contexts, list)
