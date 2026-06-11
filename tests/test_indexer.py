from unittest.mock import patch

import pytest

from src import indexer


@pytest.fixture
def mock_chroma_ollama():
    """Mocks Chroma and Ollama so they don't need to run."""
    with (
        patch("src.indexer.OllamaEmbeddings") as mock_embed,
        patch("src.indexer.Chroma") as mock_chroma,
    ):
        yield mock_embed, mock_chroma


def test_indexer_clears_old_database_before_recreation(
    tmp_path, monkeypatch, mock_chroma_ollama
):
    mock_embed, mock_chroma = mock_chroma_ollama

    # Creates a temporary directory
    fake_chroma_dir = tmp_path / "chroma_db"
    fake_chroma_dir.mkdir()

    # Creates fakes files/folders inside the directory
    old_file = fake_chroma_dir / "old_embedding.bin"
    old_file.write_text("some old vector data")

    old_subdir = fake_chroma_dir / "old_sqlite_dir"
    old_subdir.mkdir()
    (old_subdir / "db.sqlite3").write_text("old database")

    # Creates a fake temporary source file
    fake_code_file = tmp_path / "dummy_source.py"
    fake_code_file.write_text("from constants import CHROMA_DIR")

    # Overrides the global constants for this test
    monkeypatch.setattr(indexer, "CHROMA_DIR", fake_chroma_dir)
    monkeypatch.setattr(indexer, "CHUNK_SIZE", 500)
    monkeypatch.setattr(indexer, "CHUNK_OVERLAP", 50)
    monkeypatch.setattr(indexer, "EMBED_MODEL", "mock-model")

    # Runs the indexer
    indexer.indexer(fake_code_file)

    # Verifies that the preexisting files and directories inside CHROMA_DIR were correctly removed
    assert not old_file.exists()
    assert not old_subdir.exists()

    # Verifies that the function tried to initialize the right model
    mock_embed.assert_called_once_with(model="mock-model")

    # Verifies that the function called Chroma to build the index
    mock_chroma.from_texts.assert_called_once()
