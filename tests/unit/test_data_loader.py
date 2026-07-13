import pandas as pd
import pytest

from utils.data_loader import (
    build_cleaned_document,
    clean_text,
    extract_text_from_csv,
    extract_text_from_excel,
    extract_text_from_txt,
)


class TestCleanText:
    def test_removes_bom(self):
        assert clean_text("﻿Hello") == "Hello"

    def test_collapses_multiple_spaces(self):
        assert clean_text("a    b") == "a b"

    def test_collapses_multiple_blank_lines(self):
        assert clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_control_chars(self):
        assert clean_text("a\x00\x01b") == "ab"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("   a   ") == "a"

    def test_preserves_newlines_and_tabs(self):
        assert clean_text("a\nb\tc") == "a\nb\tc"


class TestBuildCleanedDocument:
    def test_valid_document_is_cleaned(self):
        doc = build_cleaned_document(
            "  Hello   world  ",
            {"source": "s", "filename": "f", "category": "c", "full_path": "p"},
        )
        assert doc is not None
        assert doc.page_content == "Hello world"
        assert doc.metadata.source == "s"

    def test_empty_content_returns_none(self):
        doc = build_cleaned_document(
            "", {"source": "s", "filename": "f", "category": "c", "full_path": "p"}
        )
        assert doc is None

    def test_missing_metadata_field_returns_none(self):
        doc = build_cleaned_document("valid text", {"source": "s"})
        assert doc is None

    def test_content_that_cleans_to_blank_returns_none(self):
        doc = build_cleaned_document(
            "   ", {"source": "s", "filename": "f", "category": "c", "full_path": "p"}
        )
        assert doc is None


class TestExtractTextFromTxt:
    def test_extracts_plain_text(self, tmp_path):
        file_path = tmp_path / "note.txt"
        file_path.write_text("Bonjour\nle monde", encoding="utf-8")

        result = extract_text_from_txt(str(file_path))

        assert result == "Bonjour\nle monde"

    def test_missing_file_returns_none(self, tmp_path):
        result = extract_text_from_txt(str(tmp_path / "missing.txt"))
        assert result is None


class TestExtractTextFromCsv:
    def test_extracts_csv_content(self, tmp_path):
        file_path = tmp_path / "data.csv"
        file_path.write_text("player,pts\nJordan,30\n", encoding="utf-8")

        result = extract_text_from_csv(str(file_path))

        assert result is not None
        assert "Jordan" in result
        assert "30" in result

    def test_missing_file_returns_none(self, tmp_path):
        result = extract_text_from_csv(str(tmp_path / "missing.csv"))
        assert result is None


class TestExtractTextFromExcel:
    def test_single_sheet_returns_string(self, tmp_path):
        file_path = tmp_path / "single.xlsx"
        pd.DataFrame({"player": ["Jordan"], "pts": [30]}).to_excel(
            file_path, sheet_name="Sheet1", index=False
        )

        result = extract_text_from_excel(str(file_path))

        assert isinstance(result, str)
        assert "Jordan" in result

    def test_multiple_sheets_returns_dict(self, tmp_path):
        file_path = tmp_path / "multi.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="One", index=False)
            pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Two", index=False)

        result = extract_text_from_excel(str(file_path))

        assert isinstance(result, dict)
        assert set(result.keys()) == {"One", "Two"}

    def test_missing_file_returns_none(self, tmp_path):
        result = extract_text_from_excel(str(tmp_path / "missing.xlsx"))
        assert result is None
