import pytest
from pydantic import ValidationError

from utils.schemas import Chunk, RAGAnswer, RAGQuery


class TestRAGQuery:
    def test_valid_question(self):
        query = RAGQuery(question="Quel joueur a le plus de points ?")
        assert query.question == "Quel joueur a le plus de points ?"

    def test_strips_whitespace(self):
        query = RAGQuery(question="  Question ?  ")
        assert query.question == "Question ?"

    def test_blank_question_rejected(self):
        with pytest.raises(ValidationError):
            RAGQuery(question="   ")

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            RAGQuery(question="")

    def test_too_long_question_rejected(self):
        with pytest.raises(ValidationError):
            RAGQuery(question="a" * 2001)


class TestRAGAnswer:
    def test_valid_answer(self):
        answer = RAGAnswer(answer="Réponse.")
        assert answer.answer == "Réponse."

    def test_empty_answer_rejected(self):
        with pytest.raises(ValidationError):
            RAGAnswer(answer="")


class TestChunk:
    def test_empty_text_rejected(self, sample_chunk_metadata):
        with pytest.raises(ValidationError):
            Chunk(id="0_0", text="", metadata=sample_chunk_metadata)

    def test_valid_chunk(self, sample_chunk_metadata):
        chunk = Chunk(id="0_0", text="some text", metadata=sample_chunk_metadata)
        assert chunk.text == "some text"
        assert chunk.metadata.chunk_id_in_doc == 0
