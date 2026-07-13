"""Tests fonctionnels du pipeline SQL (Sql_db/sql_tool.py) : génération de la
requête, garde-fous de sécurité et exécution enchaînés via
query_nba_database(), avec le LLM et la base de données mockés en frontière."""
from unittest.mock import MagicMock, patch

from Sql_db.sql_tool import query_nba_database


def make_fake_db(table_info="CREATE TABLE teams (...)", run_result="[('Lakers', 30)]"):
    fake_db = MagicMock()
    fake_db.get_table_info.return_value = table_info
    fake_db.run.return_value = run_result
    return fake_db


class TestQueryNbaDatabase:
    def test_happy_path_executes_generated_query(self):
        fake_db = make_fake_db()
        fake_llm = MagicMock()
        fake_llm.invoke.return_value.content = "```sql\nSELECT player, pts FROM player_stats ORDER BY pts DESC LIMIT 1;\n```"

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db), patch(
            "Sql_db.sql_tool.get_sql_llm", return_value=fake_llm
        ):
            result = query_nba_database("Quel joueur a le plus de points ?")

        assert result == "[('Lakers', 30)]"
        executed_query = fake_db.run.call_args[0][0]
        assert executed_query.startswith("SELECT player, pts FROM player_stats")

    def test_unsafe_query_is_rejected_without_execution(self):
        fake_db = make_fake_db()
        fake_llm = MagicMock()
        fake_llm.invoke.return_value.content = "DROP TABLE teams;"

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db), patch(
            "Sql_db.sql_tool.get_sql_llm", return_value=fake_llm
        ):
            result = query_nba_database("Efface tout")

        assert "n'est pas autorisée" in result or "pas autorisée" in result
        fake_db.run.assert_not_called()

    def test_row_limit_is_enforced_on_generated_query(self):
        fake_db = make_fake_db()
        fake_llm = MagicMock()
        fake_llm.invoke.return_value.content = "SELECT * FROM player_stats"

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db), patch(
            "Sql_db.sql_tool.get_sql_llm", return_value=fake_llm
        ):
            query_nba_database("Tous les joueurs")

        executed_query = fake_db.run.call_args[0][0]
        assert "LIMIT 30" in executed_query

    def test_sql_generation_failure_returns_friendly_message(self):
        with patch("Sql_db.sql_tool.get_db", side_effect=RuntimeError("db unreachable")):
            result = query_nba_database("Une question ?")

        assert "pas pu générer" in result

    def test_execution_error_returns_readable_message(self):
        fake_db = make_fake_db()
        fake_db.run.side_effect = RuntimeError("connexion refusée")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value.content = "SELECT * FROM teams"

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db), patch(
            "Sql_db.sql_tool.get_sql_llm", return_value=fake_llm
        ):
            result = query_nba_database("Une question ?")

        assert "Erreur lors de l'exécution" in result
