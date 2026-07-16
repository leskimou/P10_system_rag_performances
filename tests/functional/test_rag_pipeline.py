"""Tests fonctionnels du pipeline RAG (utils/chatbot.py) : plusieurs fonctions
enchaînées (search_context -> generate_answer -> ask_with_context), avec le
vector store et l'agent LLM mockés en frontière (pas d'appel réseau réel)."""
from unittest.mock import MagicMock, patch

from utils.chatbot import ask_with_context
from utils.schemas import SearchResult


def make_search_result(text: str, source: str) -> SearchResult:
    return SearchResult(
        score=90.0,
        raw_score=0.9,
        text=text,
        metadata={
            "source": source,
            "filename": source,
            "category": "root",
            "full_path": f"/tmp/{source}",
            "chunk_id_in_doc": 0,
            "start_index": 0,
        },
    )


class TestAskWithContext:
    def test_happy_path_returns_answer_and_contexts(self):
        fake_manager = MagicMock()
        fake_manager.search.return_value = [
            make_search_result("Les Lakers ont gagné.", "reddit1.pdf"),
            make_search_result("Analyse des stats.", "reddit2.pdf"),
        ]

        fake_agent = MagicMock()
        fake_agent.run_sync.return_value.output.answer = "Les Lakers sont en tête."
        fake_agent.run_sync.return_value.all_messages.return_value = []

        with patch("utils.chatbot.get_vector_store_manager", return_value=fake_manager), patch(
            "utils.chatbot.get_agent", return_value=fake_agent
        ):
            answer, contexts = ask_with_context("Qui est en tête ?", k=2)

        assert answer == "Les Lakers sont en tête."
        assert contexts == ["Les Lakers ont gagné.", "Analyse des stats."]
        fake_manager.search.assert_called_once_with("Qui est en tête ?", k=2)

    def test_sql_tool_result_is_appended_to_contexts(self):
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        fake_manager = MagicMock()
        fake_manager.search.return_value = [make_search_result("Discussion hors-sujet.", "reddit1.pdf")]

        fake_agent = MagicMock()
        fake_agent.run_sync.return_value.output.answer = "Jokić a 29.6 points par match."
        fake_agent.run_sync.return_value.all_messages.return_value = [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="query_nba_stats", content="Jokić: 29.6 pts"),
                ]
            )
        ]

        with patch("utils.chatbot.get_vector_store_manager", return_value=fake_manager), patch(
            "utils.chatbot.get_agent", return_value=fake_agent
        ):
            answer, contexts = ask_with_context("Stats de Jokić ?")

        assert answer == "Jokić a 29.6 points par match."
        assert contexts == ["Discussion hors-sujet.", "Jokić: 29.6 pts"]

    def test_blank_question_still_returns_retrieved_contexts(self):
        fake_manager = MagicMock()
        fake_manager.search.return_value = [make_search_result("contexte", "doc.pdf")]

        with patch("utils.chatbot.get_vector_store_manager", return_value=fake_manager), patch(
            "utils.chatbot.get_agent"
        ) as mock_get_agent:
            answer, contexts = ask_with_context("   ")

        assert answer == "Désolé, votre question n'a pas pu être traitée."
        assert contexts == ["contexte"]
        mock_get_agent.assert_not_called()

    def test_no_search_results_still_generates_answer(self):
        fake_manager = MagicMock()
        fake_manager.search.return_value = []

        fake_agent = MagicMock()
        fake_agent.run_sync.return_value.output.answer = "Réponse sans contexte."
        fake_agent.run_sync.return_value.all_messages.return_value = []

        with patch("utils.chatbot.get_vector_store_manager", return_value=fake_manager), patch(
            "utils.chatbot.get_agent", return_value=fake_agent
        ):
            answer, contexts = ask_with_context("Une question ?")

        assert answer == "Réponse sans contexte."
        assert contexts == []
