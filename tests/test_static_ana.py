import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas import Candidate
from src.static_ana import _rag, _rag_worker, static


# ==========================================
# 1. Tests for _rag_worker (Async)
# ==========================================
@pytest.mark.asyncio
async def test_rag_worker_success():
    """Verifies that the worker invokes the retriever and packs the updated candidate context."""
    # 1. Setup mock structures matching Pydantic expectations
    mock_smell = MagicMock()
    mock_smell.snippet = "def insecure_function(): pass"

    mock_copied_smell = MagicMock()
    mock_copied_smell.model_dump.return_value = {
        "line": 10,
        "snippet": "def insecure_function(): pass",
        "context": "doc1\n\ndoc2",
    }
    mock_smell.model_copy.return_value = mock_copied_smell

    mock_candidate = MagicMock()
    mock_candidate.smell_type = "Security Vulnerability"
    mock_candidate.smell = mock_smell

    # 2. Mock LangChain Retriever
    mock_retriever = AsyncMock()
    doc1 = MagicMock(page_content="doc1")
    doc2 = MagicMock(page_content="doc2")
    mock_retriever.ainvoke.return_value = [doc1, doc2]

    # 3. Execution dependencies
    semaphore = asyncio.Semaphore(1)
    code_path = Path("/mock/project")

    # 4. Invoke the async worker
    smell_type, smell_dict = await _rag_worker(
        candidate=mock_candidate,
        _code_path=code_path,
        semaphore=semaphore,
        retriever=mock_retriever,
    )

    # 5. Assertions
    assert smell_type == "Security Vulnerability"
    assert smell_dict["context"] == "doc1\n\ndoc2"
    mock_retriever.ainvoke.assert_called_once_with("def insecure_function(): pass")
    mock_smell.model_copy.assert_called_once_with(update={"context": "doc1\n\ndoc2"})


# ==========================================
# 2. Tests for _rag (Async Orchestrator)
# ==========================================
@pytest.mark.asyncio
@patch("src.static_ana.OllamaEmbeddings")
@patch("src.static_ana.QdrantClient")
@patch("src.static_ana.QdrantVectorStore")
@patch("src.static_ana._rag_worker", new_callable=AsyncMock)
async def test_rag_orchestration(mock_rag_worker, mock_qdrant_cls, mock_qdrant_client_cls, mock_embeddings_cls):
    """Ensures RAG initializes LangChain objects properly and gathers worker tasks."""
    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_db.as_retriever.return_value = mock_retriever
    mock_qdrant_cls.return_value = mock_db

    mock_client = MagicMock()
    mock_qdrant_client_cls.return_value = mock_client

    mock_rag_worker.return_value = ("TypeA", {"data": "info"})

    # 1. Provide a spec to the mocks so they mimic Candidate
    mock_candidate_1 = MagicMock(spec=Candidate)
    mock_candidate_2 = MagicMock(spec=Candidate)

    # 2. Explicitly cast the list to List[Candidate] to satisfy Pyright/Mypy
    candidates = cast(list[Candidate], [mock_candidate_1, mock_candidate_2])
    code_path = Path("/mock/project")

    # Execute
    results = await _rag(candidates, code_path)

    # Assertions
    mock_embeddings_cls.assert_called_once()
    mock_qdrant_client_cls.assert_called_once()
    mock_qdrant_cls.assert_called_once_with(
        client=mock_client, embedding=mock_embeddings_cls.return_value, collection_name="index"
    )
    mock_db.as_retriever.assert_called_once()
    mock_client.close.assert_called_once()

    assert mock_rag_worker.call_count == 2
    assert len(results) == 2


# ==========================================
# 3. Tests for static (Sync Orchestrator)
# ==========================================
@patch("src.static_ana.ProcessPoolExecutor")
@patch("src.static_ana._json_parser")
@patch("src.static_ana._rag", new_callable=AsyncMock)
@patch("src.static_ana.indexer")
def test_static_pipeline_orchestration(mock_indexer, mock_rag, mock_json_parser, mock_executor_cls):
    """Verifies process tracking, parsing, and execution order inside the main pipeline."""
    # 1. Mock ProcessPoolExecutor interactions
    mock_executor = MagicMock()
    mock_future = MagicMock()
    mock_executor.submit.return_value = mock_future
    mock_executor_cls.return_value = mock_executor

    # 2. Mock internal structural returns
    mock_json_parser.return_value = ["mock_cand_1", "mock_cand_2"]

    # Note: Even though static() uses asyncio.run(), mocking _rag with AsyncMock
    # allows asyncio.run to execute it like a real coroutine.
    mock_rag.return_value = [("TypeA", {"res": 1}), ("TypeB", {"res": 2})]

    # 3. Execute
    code_path = Path("/mock/project")
    pipeline_output = static(code_path)

    # 4. Assertions for background indexing
    mock_executor_cls.assert_called_once_with(max_workers=1)
    mock_executor.submit.assert_called_once_with(mock_indexer, code_path)
    mock_future.result.assert_called_once()  # Assures we blocked until indexer finished
    mock_executor.shutdown.assert_called_once_with(wait=True)

    mock_json_parser.assert_called_once_with({})
    mock_rag.assert_called_once_with(["mock_cand_1", "mock_cand_2"], code_path)
    assert pipeline_output == [("TypeA", {"res": 1}), ("TypeB", {"res": 2})]


@pytest.mark.asyncio
@patch("src.static_ana.OllamaEmbeddings")
@patch("src.static_ana.QdrantClient")
@patch("src.static_ana.QdrantVectorStore")
async def test_rag_retriever_failure(mock_qdrant_cls, mock_qdrant_client_cls, mock_embeddings_cls):
    """Ensures exceptions from the retriever are propagated cleanly out of _rag."""
    mock_db = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.ainvoke.side_effect = RuntimeError("Ollama connection failed")
    mock_db.as_retriever.return_value = mock_retriever
    mock_qdrant_cls.return_value = mock_db

    mock_client = MagicMock()
    mock_qdrant_client_cls.return_value = mock_client

    mock_candidate = MagicMock()
    mock_candidate.smell.snippet = "some code"

    with pytest.raises(RuntimeError, match="Ollama connection failed"):
        await _rag([mock_candidate], Path("/mock/project"))
