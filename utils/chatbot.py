# utils/chatbot.py
"""Pipeline RAG (recherche + génération) réutilisable hors Streamlit.

Centralise la logique utilisée par MistralChat.py afin que les scripts
d'évaluation (ans_cont_recup_ragas.py, evaluate_ragas.py) puissent appeler le
même pipeline sans dépendre de l'UI.
"""
from typing import Dict, List, Tuple

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from .config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from .vector_store import VectorStoreManager

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en t'appuyant sur le contexte fourni.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

_client: MistralClient | None = None
_vector_store_manager: VectorStoreManager | None = None


def get_client() -> MistralClient:
    global _client
    if _client is None:
        _client = MistralClient(api_key=MISTRAL_API_KEY)
    return _client


def get_vector_store_manager() -> VectorStoreManager:
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


def format_context(search_results: List[Dict[str, any]]) -> str:
    if not search_results:
        return "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
    return "\n\n---\n\n".join(
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
        for res in search_results
    )


def search_context(question: str, k: int = SEARCH_K) -> List[Dict[str, any]]:
    return get_vector_store_manager().search(question, k=k)


def generate_answer(question: str, search_results: List[Dict[str, any]]) -> str:
    final_prompt = SYSTEM_PROMPT.format(
        context_str=format_context(search_results), question=question
    )
    response = get_client().chat(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content=final_prompt)],
        temperature=0.1,
    )
    if response.choices:
        return response.choices[0].message.content
    return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."


def ask_with_context(question: str, k: int = SEARCH_K) -> Tuple[str, List[str]]:
    """Exécute le pipeline RAG complet pour une question.

    Retourne la réponse générée et la liste des textes de contexte récupérés,
    au format attendu par les datasets d'évaluation RAGAS.
    """
    search_results = search_context(question, k=k)
    answer = generate_answer(question, search_results)
    contexts = [res["text"] for res in search_results]
    return answer, contexts
