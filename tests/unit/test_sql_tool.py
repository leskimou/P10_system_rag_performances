from unittest.mock import MagicMock, patch

from Sql_db.sql_tool import (
    build_sql_generation_prompt,
    cast_round_args_to_numeric,
    clean_sql_query,
    enforce_row_limit,
    execute_sql_query,
    is_safe_select,
    normalize_union_parentheses,
    strip_redundant_outer_select,
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

    def test_parenthesized_select_is_safe(self):
        assert is_safe_select("(SELECT * FROM teams) UNION ALL (SELECT * FROM teams)") is True


class TestNormalizeUnionParentheses:
    def test_no_union_left_untouched(self):
        query = "SELECT player, pts FROM player_stats ORDER BY pts DESC LIMIT 1"
        assert normalize_union_parentheses(query) == query

    def test_wraps_unparenthesized_union_all_arms_with_order_by_limit(self):
        query = (
            "SELECT 'Top 5 scoring' AS category, t.team_name, "
            "SUM(ps.pts)::float / SUM(ps.gp) AS pts_per_game "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY pts_per_game DESC LIMIT 5 "
            "UNION ALL "
            "SELECT 'Bottom 5 scoring' AS category, t.team_name, "
            "SUM(ps.pts)::float / SUM(ps.gp) AS pts_per_game "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY pts_per_game ASC LIMIT 5"
        )
        result = normalize_union_parentheses(query)
        assert result.startswith("(SELECT 'Top 5 scoring'")
        assert "LIMIT 5) UNION ALL (SELECT 'Bottom 5 scoring'" in result
        assert result.endswith("LIMIT 5)")

    def test_wraps_multiple_chained_union_all_arms_across_newlines(self):
        query = (
            "SELECT 'Top 5 scoring' AS category, t.team_name, value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value DESC LIMIT 5\n"
            "UNION ALL\n"
            "SELECT 'Bottom 5 scoring' AS category, t.team_name, value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value ASC LIMIT 5\n"
            "UNION ALL\n"
            "SELECT 'Top 5 assists' AS category, t.team_name, value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value DESC LIMIT 5"
        )
        result = normalize_union_parentheses(query)
        assert result.count("UNION ALL") == 2
        assert result.count("(SELECT") == 3
        assert "\n" not in result

    def test_already_parenthesized_union_left_untouched(self):
        query = (
            "(SELECT 'Top 3 scoring' AS category, t.team_name, value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value DESC LIMIT 3) "
            "UNION ALL "
            "(SELECT 'Bottom 3 scoring' AS category, t.team_name, value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value ASC LIMIT 3)"
        )
        assert normalize_union_parentheses(query) == query

    def test_union_arms_without_order_by_or_limit_left_unwrapped(self):
        query = "SELECT team_code FROM teams UNION SELECT team_code FROM teams"
        assert normalize_union_parentheses(query) == query

    def test_string_literal_containing_parenthesis_does_not_break_depth_tracking(self):
        query = (
            "SELECT 'Top 5 (scoring)' AS category, pts FROM player_stats "
            "ORDER BY pts DESC LIMIT 5 "
            "UNION ALL "
            "SELECT 'Bottom 5 (scoring)' AS category, pts FROM player_stats "
            "ORDER BY pts ASC LIMIT 5"
        )
        result = normalize_union_parentheses(query)
        assert result.startswith("(SELECT 'Top 5 (scoring)'")
        assert result.endswith("LIMIT 5)")


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


class TestStripRedundantOuterSelect:
    def test_strips_extra_select_before_parenthesized_union_arms(self):
        query = (
            "SELECT (SELECT 'Top 5 rebonds' AS category, t.team_name, "
            "SUM(ps.reb)::float / SUM(ps.gp) AS value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value DESC LIMIT 5) "
            "UNION ALL "
            "(SELECT 'Bottom 5 rebonds' AS category, t.team_name, "
            "SUM(ps.reb)::float / SUM(ps.gp) AS value "
            "FROM player_stats ps JOIN teams t ON ps.team = t.team_code "
            "GROUP BY t.team_name ORDER BY value ASC LIMIT 5)"
        )
        result = strip_redundant_outer_select(query)
        assert result.startswith("(SELECT 'Top 5 rebonds'")
        assert not result.startswith("SELECT (SELECT")

    def test_leaves_single_scalar_subquery_column_untouched(self):
        query = "SELECT (SELECT MAX(pts) FROM player_stats) AS max_pts"
        assert strip_redundant_outer_select(query) == query

    def test_leaves_multiple_scalar_subquery_columns_untouched(self):
        query = (
            "SELECT (SELECT STRING_AGG(team_name, ', ') FROM teams) AS top_5, "
            "(SELECT STRING_AGG(team_name, ', ') FROM teams) AS bottom_5"
        )
        assert strip_redundant_outer_select(query) == query

    def test_leaves_query_without_leading_select_paren_untouched(self):
        query = "SELECT player, pts FROM player_stats ORDER BY pts DESC LIMIT 1"
        assert strip_redundant_outer_select(query) == query


class TestCastRoundArgsToNumeric:
    def test_casts_double_precision_expression_to_numeric(self):
        query = (
            "SELECT ROUND(SUM(ps.ast)::float / SUM(ps.gp), 1) AS avg_ast "
            "FROM player_stats ps"
        )
        result = cast_round_args_to_numeric(query)
        assert "ROUND((SUM(ps.ast)::float / SUM(ps.gp))::numeric, 1)" in result

    def test_leaves_already_cast_numeric_untouched(self):
        query = "SELECT ROUND((SUM(ps.ast)::float / SUM(ps.gp))::numeric, 1) FROM player_stats ps"
        assert cast_round_args_to_numeric(query) == query

    def test_leaves_single_arg_round_untouched(self):
        query = "SELECT ROUND(pts) FROM player_stats"
        assert cast_round_args_to_numeric(query) == query

    def test_handles_multiple_round_calls_in_same_query(self):
        query = (
            "SELECT ROUND(SUM(ps.ast)::float / SUM(ps.gp), 1) AS a, "
            "ROUND(SUM(ps.reb)::float / SUM(ps.gp), 2) AS b FROM player_stats ps"
        )
        result = cast_round_args_to_numeric(query)
        assert result.count("::numeric") == 2


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
