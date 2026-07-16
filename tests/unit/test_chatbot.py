from unittest.mock import MagicMock, patch

from utils.chatbot import format_context, generate_answer
from utils.schemas import SearchResult


def make_search_result(text="some context", source="doc.txt", score=87.654) -> SearchResult:
    return SearchResult(
        score=score,
        raw_score=score / 100,
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


class TestFormatContext:
    def test_empty_results_returns_placeholder(self):
        result = format_context([])
        assert "Aucune information pertinente" in result

    def test_formats_source_score_and_text(self):
        result = format_context([make_search_result(text="Le texte", source="doc.txt", score=87.654)])
        assert "doc.txt" in result
        assert "87.7%" in result
        assert "Le texte" in result

    def test_joins_multiple_results(self):
        result = format_context(
            [make_search_result(source="a.txt"), make_search_result(source="b.txt")]
        )
        assert "a.txt" in result
        assert "b.txt" in result
        assert result.count("Source:") == 2


class TestGenerateAnswer:
    def test_blank_question_returns_fallback_without_calling_agent(self):
        with patch("utils.chatbot.get_agent") as mock_get_agent:
            answer, tool_contexts = generate_answer("   ", [])

        assert answer == "Désolé, votre question n'a pas pu être traitée."
        assert tool_contexts == []
        mock_get_agent.assert_not_called()

    def test_happy_path_returns_agent_output(self):
        fake_agent = MagicMock()
        fake_agent.run_sync.return_value.output.answer = "Réponse générée"
        fake_agent.run_sync.return_value.all_messages.return_value = []

        with patch("utils.chatbot.get_agent", return_value=fake_agent):
            answer, tool_contexts = generate_answer("Une question ?", [])

        assert answer == "Réponse générée"
        assert tool_contexts == []
        fake_agent.run_sync.assert_called_once()

    def test_happy_path_includes_sql_tool_result_in_contexts(self):
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        fake_agent = MagicMock()
        fake_agent.run_sync.return_value.output.answer = "Réponse générée"
        fake_agent.run_sync.return_value.all_messages.return_value = [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="query_nba_stats", content="Jokić: 29.6 pts"),
                ]
            )
        ]

        with patch("utils.chatbot.get_agent", return_value=fake_agent):
            answer, tool_contexts = generate_answer("Une question ?", [])

        assert answer == "Réponse générée"
        assert tool_contexts == ["Jokić: 29.6 pts"]

    def test_agent_exception_returns_fallback(self):
        fake_agent = MagicMock()
        fake_agent.run_sync.side_effect = RuntimeError("boom")

        with patch("utils.chatbot.get_agent", return_value=fake_agent):
            answer, tool_contexts = generate_answer("Une question ?", [])

        assert answer == "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
        assert tool_contexts == []
