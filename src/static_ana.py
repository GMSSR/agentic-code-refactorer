import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings

from constants import CHROMA_DIR, EMBED_MODEL, MAX_ASYNC_WORKERS, TOP_K

from .indexer import indexer
from .schemas import Candidate


def _json_parser(_static_json: dict[str, Any]) -> list[Candidate]:
    """
    Receives the json of the static analysis tool, parses it,
    and validates it into a list of Pydantic Candidate objects.
    """
    # Placeholder simulation of parsing tool output
    parsed_candidates = []

    # Example mapping logic:
    # for item in static_json.get("issues", []):
    #     parsed_candidates.append(
    #         Candidate(
    #             smell_type=item["rule"],
    #             smell=SmellCode(line=item["line_no"], snippet=item["src"])
    #         )
    #     )

    return parsed_candidates


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


async def _rag(
    candidates: list[Candidate], code_path: Path
) -> list[tuple[str, dict[str, Any]]]:
    limit = asyncio.Semaphore(MAX_ASYNC_WORKERS)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": TOP_K, "score_threshold": 0.55}
    )

    tasks = [
        _rag_worker(candidate, code_path, limit, retriever=retriever)
        for candidate in candidates
    ]
    output = await asyncio.gather(*tasks)

    return output


def static(code_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """
    Main orchestrator for static analysis stage.
    Returns a list of tuples (smell_type, smell_object) to match main loop unpacking.
    """

    # 1. Run indexer on the background to minimize the waiting for it to finish

    executor = ProcessPoolExecutor(max_workers=1)
    indexer_future = executor.submit(indexer, code_path)

    # 2. Run static analysis tool placeholder
    tool_return = {}

    # 3. Parse into Pydantic models
    candidates = _json_parser(tool_return)

    # 4. Ensures the indexing has ended
    indexer_future.result()
    executor.shutdown(wait=True)

    # 6. Retrieves RAG context using candidate attributes
    smell_candidates = asyncio.run(_rag(candidates, code_path))

    return smell_candidates
