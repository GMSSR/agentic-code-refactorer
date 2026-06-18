import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from constants import EMBED_MODEL, INDEX_DIR, MAX_ASYNC_WORKERS, TOP_K
from src.language import get_tool

from .indexer import indexer
from .schemas import Candidate


async def _rag_worker(
    candidate: Candidate,
    _code_path: Path,
    semaphore: asyncio.Semaphore,
    retriever: BaseRetriever,
) -> tuple[str, dict[str, Any]]:
    """Retrieves relevant surrounding code or documentation context."""

    async with semaphore:
        docs = await retriever.ainvoke(candidate.smell.snippet)
        context_code = "\n\n".join(doc.page_content for doc in docs)
        smell = candidate.smell.model_copy(update={"context": context_code})

        return (candidate.smell_type, smell.model_dump())


async def _rag(candidates: list[Candidate], code_path: Path) -> list[tuple[str, dict[str, Any]]]:
    limit = asyncio.Semaphore(MAX_ASYNC_WORKERS)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    client = QdrantClient(path=str(INDEX_DIR))
    try:
        db = QdrantVectorStore(client=client, embedding=embeddings, collection_name="index")
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "score_threshold": 0.55})

        tasks = [_rag_worker(candidate, code_path, limit, retriever=retriever) for candidate in candidates]
        output = await asyncio.gather(*tasks)

        return output
    finally:
        client.close()


def static(code_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """
    Main orchestrator for static analysis stage.
    Returns a list of tuples (smell_type, smell_object) to match main loop unpacking.
    """

    # 1. Run indexer on the background to minimize the waiting for it to finish

    executor = ProcessPoolExecutor(max_workers=1)
    indexer_future = executor.submit(indexer, code_path)

    # 2. Run static analysis tool
    candidates = get_tool(code_path)

    # 3. Ensures the indexing has ended
    indexer_future.result()
    executor.shutdown(wait=True)

    # 4. Retrieves RAG context using candidate attributes
    smell_candidates = asyncio.run(_rag(candidates, code_path))

    return smell_candidates
