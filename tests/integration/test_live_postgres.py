"""Tests d'intégration : requêtes réelles (lecture seule) sur la base
PostgreSQL des statistiques NBA.

Auto-skippés si POSTGRES_URL n'est pas configuré (DB_HOST/DB_PORT/DB_NAME/
DB_USER/DB_PWD absents du .env)."""
import pytest

from utils.config import POSTGRES_URL

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL non configuré"),
]


def test_get_db_introspects_real_schema():
    from Sql_db.sql_tool import get_db

    db = get_db()

    table_info = db.get_table_info()

    assert "teams" in table_info
    assert "player_stats" in table_info


def test_execute_sql_query_reads_teams_table():
    from Sql_db.sql_tool import execute_sql_query

    result = execute_sql_query("SELECT COUNT(*) FROM teams LIMIT 30")

    assert "Erreur" not in result
