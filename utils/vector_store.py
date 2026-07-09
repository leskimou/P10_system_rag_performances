# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import logfire
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Utilisé pour le format attendu par le splitter

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)
from .schemas import Chunk, ChunkMetadata, CleanedDocument, EmbeddedChunk, RAGQuery, SearchResult

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Chunk] = []
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents: List[CleanedDocument]) -> List[Chunk]:
        """Découpe les documents en chunks validés avec métadonnées."""
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        with logfire.span("split_documents_to_chunks", num_documents=len(documents)):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )

            all_chunks: List[Chunk] = []
            for doc_counter, doc in enumerate(documents):
                langchain_doc = Document(page_content=doc.page_content, metadata=doc.metadata.model_dump())
                split_docs = text_splitter.split_documents([langchain_doc])
                logging.info(f"  Document '{doc.metadata.filename}' découpé en {len(split_docs)} chunks.")

                for i, split_doc in enumerate(split_docs):
                    chunk_metadata = ChunkMetadata(
                        **doc.metadata.model_dump(),
                        chunk_id_in_doc=i,
                        start_index=split_doc.metadata.get("start_index", -1),
                    )
                    all_chunks.append(Chunk(id=f"{doc_counter}_{i}", text=split_doc.page_content, metadata=chunk_metadata))

            logging.info(f"Total de {len(all_chunks)} chunks créés.")
            return all_chunks

    def _generate_embedded_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral.

        Un lot dont l'appel API échoue ou dont la réponse ne correspond pas
        au nombre de chunks envoyés est ignoré (les chunks concernés ne
        seront pas indexés), plutôt que de bloquer toute l'indexation.
        """
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return []
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return []

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        with logfire.span("generate_embedded_chunks", num_chunks=len(chunks)):
            embedded_chunks: List[EmbeddedChunk] = []
            total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
                batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
                texts_to_embed = [chunk.text for chunk in batch_chunks]

                logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
                try:
                    response = self.mistral_client.embeddings.create(model=EMBEDDING_MODEL, inputs=texts_to_embed)
                except SDKError as e:
                    logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
                    continue

                if len(response.data) != len(batch_chunks):
                    logging.error(
                        f"Lot {batch_num}: {len(response.data)} embeddings reçus pour {len(batch_chunks)} chunks envoyés. Lot ignoré."
                    )
                    continue

                for chunk, data in zip(batch_chunks, response.data):
                    embedded_chunks.append(EmbeddedChunk(chunk=chunk, vector=data.embedding))

            logging.info(f"{len(embedded_chunks)}/{len(chunks)} chunks ont reçu un embedding.")
            return embedded_chunks

    def build_index(self, documents: List[CleanedDocument]):
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        chunks = self._split_documents_to_chunks(documents)
        if not chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        embedded_chunks = self._generate_embedded_chunks(chunks)
        if not embedded_chunks:
            logging.error("Aucun embedding n'a pu être généré. Annulation de l'indexation.")
            self.document_chunks = []
            self.index = None
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
            return

        self.document_chunks = [ec.chunk for ec in embedded_chunks]
        vectors = np.array([ec.vector for ec in embedded_chunks], dtype="float32")

        dimension = vectors.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: Optional[float] = None) -> List[SearchResult]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des SearchResult pertinents, triés par score décroissant
        """
        try:
            query = RAGQuery(question=query_text)
        except Exception as e:
            logging.error(f"Requête invalide: {e}")
            return []

        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
            return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query.question}'")
        try:
            response = self.mistral_client.embeddings.create(model=EMBEDDING_MODEL, inputs=[query.question])
        except SDKError as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            return []

        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)

        search_k = k * 3 if min_score is not None else k
        scores, indices = self.index.search(query_embedding, search_k)

        results: List[SearchResult] = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    raw_score = float(scores[0][i])
                    similarity = raw_score * 100

                    min_score_percent = min_score * 100 if min_score is not None else 0
                    if min_score is not None and similarity < min_score_percent:
                        logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                        continue

                    results.append(SearchResult(score=similarity, raw_score=raw_score, text=chunk.text, metadata=chunk.metadata))
                else:
                    logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

        results.sort(key=lambda r: r.score, reverse=True)
        if len(results) > k:
            results = results[:k]

        logging.info(f"{len(results)} chunks pertinents trouvés.")
        return results