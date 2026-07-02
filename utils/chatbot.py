# utils/chatbot.py
"""Pipeline RAG (recherche + génération) réutilisable hors Streamlit.

Centralise la logique utilisée par MistralChat.py afin que les scripts
d'évaluation (ans_cont_recup_ragas.py, evaluate_ragas.py) puissent appeler le
même pipeline sans dépendre de l'UI.
"""
import logging
from typing import List, Tuple

import logfire
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

from .config import LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P, MODEL_NAME, SEARCH_K
from .schemas import RAGAnswer, RAGQuery, SearchResult
from .vector_store import VectorStoreManager

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en t'appuyant sur le contexte fourni.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

_agent: Agent | None = None
_vector_store_manager: VectorStoreManager | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(MistralModel(MODEL_NAME), output_type=RAGAnswer)
    return _agent


def get_vector_store_manager() -> VectorStoreManager:
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


def format_context(search_results: List[SearchResult]) -> str:
    if not search_results:
        return "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
    return "\n\n---\n\n".join(
        f"Source: {res.metadata.source} (Score: {res.score:.1f}%)\nContenu: {res.text}"
        for res in search_results
    )


def search_context(question: str, k: int = SEARCH_K) -> List[SearchResult]:
    with logfire.span("search_context", question=question, k=k):
        return get_vector_store_manager().search(question, k=k)


def generate_answer(question: str, search_results: List[SearchResult]) -> str:
    try:
        query = RAGQuery(question=question)
    except ValidationError as e:
        logging.error(f"Question invalide, réponse non générée: {e}")
        return "Désolé, votre question n'a pas pu être traitée."

    final_prompt = SYSTEM_PROMPT.format(
        context_str=format_context(search_results), question=query.question
    )
    try:
        with logfire.span("generate_answer", question=query.question):
            result = get_agent().run_sync(
                final_prompt,
                model_settings={
                    "temperature": LLM_TEMPERATURE,
                    "top_p": LLM_TOP_P,
                    "max_tokens": LLM_MAX_TOKENS,
                },
            )
        return result.output.answer
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse par l'agent: {e}")
        return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."


def ask_with_context(question: str, k: int = SEARCH_K) -> Tuple[str, List[str]]:
    """Exécute le pipeline RAG complet pour une question.

    Retourne la réponse générée et la liste des textes de contexte récupérés,
    au format attendu par les datasets d'évaluation RAGAS.
    """
    with logfire.span("ask_with_context", question=question):
        search_results = search_context(question, k=k)
        answer = generate_answer(question, search_results)
        contexts = [res.text for res in search_results]
        return answer, contexts
