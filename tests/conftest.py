"""Fixtures partagées pour tous les tests (unitaires, fonctionnels, intégration)."""
import os

# doit être défini avant l'import de faiss/torch (via easyocr) : sur Windows, ces deux
# libs embarquent chacune leur propre runtime OpenMP, ce qui plante le process
# (Fatal Python error: Aborted) dès que les deux sont chargées dans la même session.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from utils.schemas import ChunkMetadata, DocumentMetadata


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        source="doc.txt", filename="doc.txt", category="root", full_path="/tmp/doc.txt"
    )


@pytest.fixture
def sample_chunk_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        source="doc.txt",
        filename="doc.txt",
        category="root",
        full_path="/tmp/doc.txt",
        chunk_id_in_doc=0,
        start_index=0,
    )
