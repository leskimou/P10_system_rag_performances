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
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.mistral import MistralModel

from .config import LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P, MODEL_NAME, SEARCH_K
from .schemas import RAGAnswer, RAGQuery, SearchResult
from Sql_db.sql_tool import query_nba_database
from .vector_store import VectorStoreManager

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un data analyst expert qui appuie les coachs et le staff
technique d'une équipe NBA dans leur préparation de match.
Ta mission est de fournir des analyses statistiques exploitables en t'appuyant sur le contexte
fourni. Mets en avant les chiffres clés. Reste synthétique.

Le contexte ci-dessous provient de documents généraux (description des colonnes, commentaires,
liste des équipes) : il ne contient PAS les statistiques chiffrées des joueurs ou des équipes.
Ne conclus jamais qu'une statistique est indisponible sur la seule base de ce contexte.

Dès que la question porte sur une statistique chiffrée (classement, moyenne, total, comparaison
de stats de joueurs ou d'équipes, y compris des stats moins courantes comme les contres/blocks,
interceptions, pertes de balle, etc.), tu DOIS appeler l'outil `query_nba_stats` pour interroger
la base de données NBA avant de répondre, même si cette statistique n'apparaît pas dans le
contexte ci-dessous. N'affirme qu'une donnée n'existe pas qu'après avoir appelé cet outil et
constaté que son résultat est vide ou en erreur.

Si pertinent, utilise aussi les commentaires du contexte pour enrichir ta réponse.

Si la question ne concerne pas le basketball ou les statistiques NBA, réponds poliment que tu ne
peux pas répondre à cette question.

---
{context_str}
---

QUESTION DU STAFF:
{question}

ANALYSE DATA:"""

_agent: Agent | None = None
_vector_store_manager: VectorStoreManager | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(MistralModel(MODEL_NAME), output_type=RAGAnswer)

        @_agent.tool_plain
        def query_nba_stats(question: str) -> str:
            """Interroge la base de données PostgreSQL des statistiques NBA (joueurs et
            équipes) pour répondre à une question chiffrée : classements, moyennes,
            comparaisons de stats, totaux, etc. À utiliser dès que la question porte sur
            des chiffres ou des statistiques précises plutôt que sur du contexte général.
            """
            return query_nba_database(question)

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


def _tool_result_contexts(result) -> List[str]:
    """Extrait le texte renvoyé par l'outil `query_nba_stats` (base SQL) sur ce run,
    afin de pouvoir l'inclure dans les `contexts` RAGAS : c'est lui, et non le
    vector store, qui a réellement servi à générer la réponse pour les questions
    chiffrées."""
    contexts = []
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart) and part.tool_name == "query_nba_stats":
                contexts.append(part.model_response_str())
    return contexts


def generate_answer(question: str, search_results: List[SearchResult]) -> Tuple[str, List[str]]:
    try:
        query = RAGQuery(question=question)
    except ValidationError as e:
        logging.error(f"Question invalide, réponse non générée: {e}")
        return "Désolé, votre question n'a pas pu être traitée.", []

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
        return result.output.answer, _tool_result_contexts(result)
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse par l'agent: {e}")
        return "Désolé, je n'ai pas pu générer de réponse valide pour le moment.", []


def ask_with_context(question: str, k: int = SEARCH_K) -> Tuple[str, List[str]]:
    """Exécute le pipeline RAG complet pour une question.

    Retourne la réponse générée et la liste des textes de contexte récupérés
    (vector store + résultats de l'outil SQL le cas échéant), au format attendu
    par les datasets d'évaluation RAGAS.
    """
    with logfire.span("ask_with_context", question=question):
        search_results = search_context(question, k=k)
        answer, tool_contexts = generate_answer(question, search_results)
        contexts = [res.text for res in search_results] + tool_contexts
        return answer, contexts
