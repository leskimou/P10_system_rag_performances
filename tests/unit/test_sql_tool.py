from unittest.mock import MagicMock, patch

from Sql_db.sql_tool import (
    build_sql_generation_prompt,
    clean_sql_query,
    enforce_row_limit,
    execute_sql_query,
    is_safe_select,
)


class TestCleanSqlQuery:
    def test_strips_sql_fence(self):
        raw = "```sql\nSELECT * FROM teams;\n```"
        assert clean_sql_query(raw) == "SELECT * FROM teams"

    def test_strips_plain_fence(self):
        raw = "```\nSELECT * FROM teams;\n```"
        assert clean_sql_query(raw) == "SELECT * FROM teams"

    def test_no_fence_still_stripped(self):
        raw = "  SELECT * FROM teams;  "
        assert clean_sql_query(raw) == "SELECT * FROM teams"

    def test_trailing_semicolon_removed_without_fence(self):
        assert clean_sql_query("SELECT 1;") == "SELECT 1"


class TestIsSafeSelect:
    def test_select_is_safe(self):
        assert is_safe_select("SELECT * FROM teams") is True

    def test_case_insensitive_select(self):
        assert is_safe_select("select player from player_stats") is True

    def test_drop_is_unsafe(self):
        assert is_safe_select("DROP TABLE teams") is False

    def test_update_is_unsafe(self):
        assert is_safe_select("UPDATE teams SET team_name = 'x'") is False

    def test_non_select_start_is_unsafe(self):
        assert is_safe_select("EXPLAIN SELECT * FROM teams") is False

    def test_select_with_forbidden_keyword_embedded_is_unsafe(self):
        assert is_safe_select("SELECT * FROM teams; DROP TABLE teams;") is False


class TestEnforceRowLimit:
    def test_adds_limit_when_missing(self):
        result = enforce_row_limit("SELECT * FROM teams", max_rows=30)
        assert result == "SELECT * FROM teams LIMIT 30"

    def test_leaves_existing_limit_untouched(self):
        result = enforce_row_limit("SELECT * FROM teams LIMIT 5", max_rows=30)
        assert result == "SELECT * FROM teams LIMIT 5"

    def test_limit_detection_is_case_insensitive(self):
        result = enforce_row_limit("SELECT * FROM teams limit 5", max_rows=30)
        assert result == "SELECT * FROM teams limit 5"


class TestExecuteSqlQuery:
    def test_requests_labeled_columns_from_db(self):
        fake_db = MagicMock()
        fake_db.run.return_value = "[{'player': 'Ivica Zubac', 'reb': 1008}]"

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db):
            result = execute_sql_query("SELECT player, reb FROM player_stats")

        fake_db.run.assert_called_once_with(
            "SELECT player, reb FROM player_stats", include_columns=True
        )
        assert result == "[{'player': 'Ivica Zubac', 'reb': 1008}]"

    def test_empty_result_returns_placeholder_message(self):
        fake_db = MagicMock()
        fake_db.run.return_value = ""

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db):
            result = execute_sql_query("SELECT 1")

        assert result == "Aucun résultat trouvé pour cette requête."

    def test_db_exception_returns_readable_error(self):
        fake_db = MagicMock()
        fake_db.run.side_effect = RuntimeError("connection refused")

        with patch("Sql_db.sql_tool.get_db", return_value=fake_db):
            result = execute_sql_query("SELECT 1")

        assert "connection refused" in result


class TestBuildSqlGenerationPrompt:
    def test_includes_question_and_schema(self):
        fake_db = MagicMock()
        fake_db.get_table_info.return_value = "CREATE TABLE teams (...)"

        prompt = build_sql_generation_prompt("Quel joueur ?", fake_db)

        assert "Quel joueur ?" in prompt
        assert "CREATE TABLE teams (...)" in prompt
