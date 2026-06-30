# utils/schemas.py
"""Modèles Pydantic utilisés à chaque étape du pipeline RAG.

Remplacent les dict non validés qui circulaient auparavant entre
data_loader.py, vector_store.py et chatbot.py.
"""
from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    source: str
    filename: str
    category: str
    full_path: str
    sheet: str | None = None


class RawDocument(BaseModel):
    """Sortie de l'extraction, avant nettoyage."""

    page_content: str = Field(min_length=1)
    metadata: DocumentMetadata


class CleanedDocument(BaseModel):
    """Sortie du nettoyage, prête pour le chunking."""

    page_content: str = Field(min_length=1)
    metadata: DocumentMetadata


class ChunkMetadata(DocumentMetadata):
    chunk_id_in_doc: int
    start_index: int


class Chunk(BaseModel):
    id: str
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    vector: list[float] = Field(min_length=1)


class SearchResult(BaseModel):
    score: float
    raw_score: float
    text: str
    metadata: ChunkMetadata


class RAGQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class RAGAnswer(BaseModel):
    """Sortie structurée de l'agent Pydantic AI du chatbot."""

    answer: str = Field(min_length=1)
