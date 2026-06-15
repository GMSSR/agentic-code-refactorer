from unittest.mock import MagicMock, patch

import pytest

from src import indexer


@pytest.fixture
def mock_qdrant_ollama():
    """Mocks Qdrant client, store and Ollama so they don't need to run."""
    with (
        patch("src.indexer.OllamaEmbeddings") as mock_embed,
        patch("src.indexer.QdrantClient") as mock_client_cls,
        patch("src.indexer.QdrantVectorStore") as mock_qdrant,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        yield mock_embed, mock_client, mock_qdrant


def test_indexer_clears_old_database_before_recreation(
    tmp_path, monkeypatch, mock_qdrant_ollama
):
    mock_embed, mock_client, mock_qdrant = mock_qdrant_ollama

    # Creates a temporary directory
    fake_index_dir = tmp_path / "index_db"
    fake_index_dir.mkdir()

    # Creates fake files/folders inside the directory
    old_file = fake_index_dir / "old_embedding.bin"
    old_file.write_text("some old vector data")

    old_subdir = fake_index_dir / "old_sqlite_dir"
    old_subdir.mkdir()
    (old_subdir / "db.sqlite3").write_text("old database")

    # Creates a fake temporary source file
    fake_code_file = tmp_path / "dummy_source.py"
    fake_code_file.write_text("from constants import INDEX_DIR")

    # Overrides the global constants for this test
    monkeypatch.setattr(indexer, "INDEX_DIR", fake_index_dir)
    monkeypatch.setattr(indexer, "CHUNK_SIZE", 500)
    monkeypatch.setattr(indexer, "CHUNK_OVERLAP", 50)
    monkeypatch.setattr(indexer, "EMBED_MODEL", "mock-model")

    # Runs the indexer
    indexer.indexer(fake_code_file)

    # Verifies that the preexisting files and directories inside INDEX_DIR were correctly removed
    assert not old_file.exists()
    assert not old_subdir.exists()

    # Verifies that the function tried to initialize the right model
    mock_embed.assert_called_once_with(model="mock-model")

    # Verifies that the function called QdrantVectorStore to build the index
    mock_qdrant.from_texts.assert_called_once_with(
        client=mock_client,
        texts=["from constants import INDEX_DIR"],
        embedding=mock_embed.return_value,
        path=str(fake_index_dir),
        collection_name="index",
    )
    mock_client.close.assert_called_once()
