"""Tests fonctionnels de l'orchestration d'indexation (indexer.py) : chargement
des documents puis construction de l'index, avec le vector store et le
chargement de fichiers mockés."""
from unittest.mock import MagicMock, patch

from indexer import run_indexing


class TestRunIndexing:
    def test_builds_index_from_loaded_documents(self):
        fake_documents = [MagicMock(), MagicMock()]
        fake_vector_store = MagicMock()

        with patch("indexer.load_and_parse_files", return_value=fake_documents) as mock_load, patch(
            "indexer.VectorStoreManager", return_value=fake_vector_store
        ):
            run_indexing(input_directory="inputs")

        mock_load.assert_called_once_with("inputs")
        fake_vector_store.build_index.assert_called_once_with(fake_documents)

    def test_no_documents_skips_index_build(self):
        with patch("indexer.load_and_parse_files", return_value=[]), patch(
            "indexer.VectorStoreManager"
        ) as mock_manager_cls:
            run_indexing(input_directory="inputs")

        mock_manager_cls.assert_not_called()

    def test_download_failure_skips_loading_and_indexing(self):
        with patch("indexer.download_and_extract_zip", return_value=False) as mock_download, patch(
            "indexer.load_and_parse_files"
        ) as mock_load, patch("indexer.VectorStoreManager") as mock_manager_cls:
            run_indexing(input_directory="inputs", data_url="http://example.test/data.zip")

        mock_download.assert_called_once_with("http://example.test/data.zip", "inputs")
        mock_load.assert_not_called()
        mock_manager_cls.assert_not_called()

    def test_successful_download_then_loads_and_indexes(self):
        fake_documents = [MagicMock()]
        fake_vector_store = MagicMock()

        with patch("indexer.download_and_extract_zip", return_value=True), patch(
            "indexer.load_and_parse_files", return_value=fake_documents
        ), patch("indexer.VectorStoreManager", return_value=fake_vector_store):
            run_indexing(input_directory="inputs", data_url="http://example.test/data.zip")

        fake_vector_store.build_index.assert_called_once_with(fake_documents)
