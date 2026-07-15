# Sql_db/sql_tool.py
"""Tool LangChain SQL : génère et exécute des requêtes SQL en lecture seule
sur la base PostgreSQL des statistiques NBA (`teams`, `player_stats`), à
partir d'une question en langage naturel.
"""
import logging
import re

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_mistralai import ChatMistralAI

from utils.config import MISTRAL_API_KEY, MODEL_NAME, POSTGRES_URL

MAX_ROWS = 30

FEW_SHOT_EXAMPLES = [
    {
        "question": "Quel joueur a marqué le plus de points ?",
        "query": "SELECT player, pts FROM player_stats ORDER BY pts DESC LIMIT 1;",
    },
    {
        "question": "Quelle est la moyenne de rebonds des joueurs du Miami Heat ?",
        "query": "SELECT AVG(reb) FROM player_stats WHERE team = 'MIA';",
    },
    {
        "question": "Combien de joueurs ont plus de 500 passes décisives ?",
        "query": "SELECT COUNT(*) FROM player_stats WHERE ast > 500;",
    },
    {
        "question": "Quel est le nom complet de l'équipe dont le code est 'OKC' ?",
        "query": "SELECT team_name FROM teams WHERE team_code = 'OKC';",
    },
    {
        "question": "Top 3 des joueurs par minutes jouées, avec le nom complet de leur équipe.",
        "query": (
            "SELECT ps.player, ps.min, t.team_name FROM player_stats ps "
            "JOIN teams t ON ps.team = t.team_code ORDER BY ps.min DESC LIMIT 3;"
        ),
    },
    {
        "question": "Quelle équipe a le meilleur pourcentage de réussite à trois points ?",
        "query": (
            "SELECT t.team_name, SUM(ps.n3pm)::float / NULLIF(SUM(ps.n3pa), 0) AS pct "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY pct DESC LIMIT 1;"
        ),
    },
    {
        "question": (
            "Quelles sont les grandes tendances de la ligue cette saison en matière de "
            "scoring, de passes et de rebonds ?"
        ),
        "query": (
            "SELECT "
            "SUM(pts)::float / SUM(gp) AS pts_per_game, "
            "SUM(reb)::float / SUM(gp) AS reb_per_game, "
            "SUM(oreb)::float / SUM(gp) AS oreb_per_game, "
            "SUM(dreb)::float / SUM(gp) AS dreb_per_game, "
            "SUM(ast)::float / SUM(gp) AS ast_per_game, "
            "SUM(fgm)::float / NULLIF(SUM(fga), 0) AS fg_pct, "
            "SUM(n3pm)::float / NULLIF(SUM(n3pa), 0) AS n3p_pct, "
            "SUM(ftm)::float / NULLIF(SUM(fta), 0) AS ft_pct "
            "FROM player_stats;"
        ),
    },
]

_EXAMPLE_PROMPT = PromptTemplate.from_template("Question: {question}\nSQLQuery: {query}")

_PREFIX = """Tu es un expert PostgreSQL. Étant donné une question en français et le schéma \
de base de données ci-dessous, écris une unique requête SQL SELECT qui répond à la question.

Règles :
- Réponds uniquement avec la requête SQL, sans explication, sans balises markdown.
- Utilise uniquement les tables et colonnes listées dans le schéma.
- N'écris jamais de requête autre que SELECT.
- Inclue toujours dans le SELECT la ou les valeurs chiffrées demandées par la question
  (pas seulement des colonnes descriptives comme un nom), pour que la réponse finale
  puisse citer le chiffre exact.
- Si la question porte sur une statistique d'équipe agrégée sur l'ensemble de ses joueurs
  (pourcentage, moyenne, total), calcule l'agrégat avec GROUP BY sur l'équipe plutôt que
  de te baser sur la statistique d'un seul joueur. Pour un pourcentage de réussite
  (tirs, lancers-francs, 3 points), calcule toujours le ratio somme(réussites) /
  somme(tentatives) sur le groupe plutôt qu'une moyenne des pourcentages individuels.

Schéma de la base de données :
{table_info}

Exemples :"""

_SUFFIX = """Question: {input}
SQLQuery:"""

_db: SQLDatabase | None = None
_sql_llm: ChatMistralAI | None = None


def get_db() -> SQLDatabase:
    """Retourne le SQLDatabase LangChain connecté à PostgreSQL (introspection dynamique
    de tout le schéma public), créé une seule fois."""
    global _db
    if _db is None:
        if not POSTGRES_URL:
            raise RuntimeError(
                "POSTGRES_URL n'est pas configuré : renseignez DB_HOST, DB_PORT, DB_NAME, "
                "DB_USER et DB_PWD dans le fichier .env."
            )
        _db = SQLDatabase.from_uri(POSTGRES_URL)
    return _db


def get_sql_llm() -> ChatMistralAI:
    """Retourne le LLM dédié à la génération SQL (temperature=0, séparé de l'agent
    pydantic_ai principal du chatbot), créé une seule fois."""
    global _sql_llm
    if _sql_llm is None:
        _sql_llm = ChatMistralAI(model=MODEL_NAME, temperature=0, mistral_api_key=MISTRAL_API_KEY)
    return _sql_llm


def build_sql_generation_prompt(question: str, db: SQLDatabase) -> str:
    """Construit le prompt few-shot de génération SQL pour une question donnée."""
    few_shot_prompt = FewShotPromptTemplate(
        examples=FEW_SHOT_EXAMPLES,
        example_prompt=_EXAMPLE_PROMPT,
        prefix=_PREFIX,
        suffix=_SUFFIX,
        input_variables=["input", "table_info"],
    )
    return few_shot_prompt.format(input=question, table_info=db.get_table_info())


_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def clean_sql_query(raw_text: str) -> str:
    """Extrait la requête SQL brute d'une réponse LLM, en tolérant d'éventuelles
    balises markdown ```sql ... ``` autour de la requête."""
    match = _SQL_FENCE_RE.search(raw_text)
    query = match.group(1) if match else raw_text
    return query.strip().rstrip(";").strip()


def generate_sql_query(question: str) -> str:
    """Génère une requête SQL pour répondre à `question`, via le LLM dédié."""
    db = get_db()
    prompt = build_sql_generation_prompt(question, db)
    response = get_sql_llm().invoke(prompt)
    return clean_sql_query(response.content)


_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE
)


def is_safe_select(query: str) -> bool:
    """True si `query` est un unique SELECT sans mot-clé de modification de données/schéma."""
    stripped = query.strip()
    if not re.match(r"^SELECT\b", stripped, re.IGNORECASE):
        return False
    if _FORBIDDEN_KEYWORDS_RE.search(stripped):
        return False
    return True


def enforce_row_limit(query: str, max_rows: int = MAX_ROWS) -> str:
    """Ajoute `LIMIT max_rows` à `query` si elle n'a pas déjà de LIMIT."""
    if re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        return query
    return f"{query.rstrip(';').strip()} LIMIT {max_rows}"


def execute_sql_query(query: str) -> str:
    """Exécute `query` sur la base et retourne le résultat formaté en texte.

    Toute erreur d'exécution (SQL invalide, colonne inexistante, DB injoignable) est
    catchée et retournée comme message lisible plutôt que de lever une exception.
    """
    try:
        # include_columns=True : renvoie des lignes {colonne: valeur} plutôt que des
        # tuples positionnels nus, pour que ce texte reste interprétable une fois sorti
        # de son contexte (utilisé tel quel comme contexte RAGAS pour faithfulness).
        result = get_db().run(query, include_columns=True)
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution de la requête SQL générée: {e}")
        return f"Erreur lors de l'exécution de la requête SQL : {e}"
    if not result:
        return "Aucun résultat trouvé pour cette requête."
    return str(result)


def query_nba_database(question: str) -> str:
    """Génère puis exécute une requête SQL en lecture seule pour répondre à une
    question chiffrée sur les statistiques NBA (joueurs et équipes).

    Retourne toujours une chaîne de texte : soit les résultats de la requête, soit
    un message d'erreur explicite. Ne lève jamais d'exception.
    """
    try:
        query = generate_sql_query(question)
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la requête SQL: {e}")
        return "Désolé, je n'ai pas pu générer de requête SQL pour cette question."

    if not is_safe_select(query):
        logging.error(f"Requête SQL générée rejetée (non-SELECT ou dangereuse): {query}")
        return "Désolé, je ne peux pas exécuter cette requête (elle n'est pas autorisée)."

    safe_query = enforce_row_limit(query)
    logging.info(f"Exécution de la requête SQL générée: {safe_query}")
    return execute_sql_query(safe_query)
