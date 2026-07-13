from Sql_db.load_excel_to_db import clean_column_name


class TestCleanColumnName:
    def test_lowercases(self):
        assert clean_column_name("Team Name") == "team_name"

    def test_percent_suffix(self):
        assert clean_column_name("FG%") == "fg_pct"

    def test_plus_minus(self):
        assert clean_column_name("+/-") == "plus_minus"

    def test_slash_becomes_per(self):
        assert clean_column_name("PTS/GP") == "pts_per_gp"

    def test_leading_digit_gets_prefixed(self):
        assert clean_column_name("3PM") == "n3pm"

    def test_non_alphanumeric_collapsed_to_underscore(self):
        assert clean_column_name("Team (Home)") == "team_home"

    def test_accepts_non_string_input(self):
        assert clean_column_name(3) == "n3"
