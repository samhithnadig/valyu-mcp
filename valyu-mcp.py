from typing import Any, Dict, List, Optional
import asyncio
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from valyu import Valyu, SearchResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

mcp = FastMCP("valyu-mcp")
valyu: Optional[Valyu] = None


async def _async_valyu_query(
    query: str,
    vertical: Optional[str] = None,
    max_num_results: int = 5,
    max_price: int = 10,
    relevance_threshold: float = 0.5,
) -> Optional[SearchResponse]:
    """
    Executes a Valyu API query inside a background thread using asyncio.to_thread
    to prevent blocking the MCP JSON-RPC event loop during concurrent requests.
    """
    if valyu is None:
        logger.error("Valyu client is not initialized.")
        return None

    try:
        logger.info(
            f"Making Valyu request - query: '{query}', vertical: {vertical}, "
            f"max_results: {max_num_results}, threshold: {relevance_threshold}"
        )

        def _call_api():
            kwargs: Dict[str, Any] = {
                "query": query,
                "max_num_results": max_num_results,
                "max_price": max_price,
            }
            if vertical:
                if hasattr(valyu, "search"):
                    try:
                        return valyu.search(data_vertical=vertical, relevance_threshold=relevance_threshold, **kwargs)
                    except TypeError:
                        pass
                return valyu.context(search_type=vertical, query_rewrite=False, **kwargs)
            else:
                return valyu.context(search_type="all", query_rewrite=False, **kwargs)

        response = await asyncio.to_thread(_call_api)
        logger.info(f"Received Valyu response with {len(getattr(response, 'results', []) or [])} results")
        return response
    except Exception as e:
        logger.error(f"Error making Valyu request: {str(e)}")
        return None


def format_valyu_results(data: Optional[SearchResponse]) -> str:
    """Formats search results into clean markdown context for LLM agents."""
    if not data or not getattr(data, "results", None):
        return "No results found."

    results = []
    for i, result in enumerate(data.results, 1):
        title = getattr(result, "title", "Untitled")
        content = getattr(result, "content", None) or getattr(result, "description", "No content available.")
        url = getattr(result, "url", "N/A")
        relevance = getattr(result, "score", None) or getattr(result, "relevance_score", None)

        score_line = f"Relevance Score: {relevance}\n" if relevance is not None else ""
        results.append(
            f"[{i}] {title}\n"
            f"{score_line}"
            f"URL: {url}\n"
            f"Content: {content.strip()}"
        )

    return "\n\n---\n\n".join(results)


# ==============================================================================
# Domain-Specific Vertical Search Tools
# ==============================================================================

@mcp.tool()
async def valyu_academic_search(
    query: str, max_results: int = 5, relevance_threshold: float = 0.55
) -> str:
    """
    Search academic literature, formal algorithm proofs, arXiv preprints, and computer science papers.
    Use for: theoretical algorithms, formal verification, ML model architectures, arXiv research.
    """
    max_results = min(max(1, max_results), 10)
    data = await _async_valyu_query(
        query=query,
        vertical="academic",
        max_num_results=max_results,
        relevance_threshold=relevance_threshold,
    )
    if not data:
        return "Unable to fetch academic context from Valyu."
    return format_valyu_results(data)


@mcp.tool()
async def valyu_biomedical_search(
    query: str, max_results: int = 5, relevance_threshold: float = 0.55
) -> str:
    """
    Search life-sciences literature, clinical trials, and PubMed citations.
    Use for: biomedical entities, clinical protocols, medical informatics, health diagnostics.
    """
    max_results = min(max(1, max_results), 10)
    data = await _async_valyu_query(
        query=query,
        vertical="biomedical",
        max_num_results=max_results,
        relevance_threshold=relevance_threshold,
    )
    if not data:
        return "Unable to fetch biomedical context from Valyu."
    return format_valyu_results(data)


@mcp.tool()
async def valyu_finance_search(
    query: str, max_results: int = 5, relevance_threshold: float = 0.60
) -> str:
    """
    Search financial filings, regulatory disclosures (SEC 10-K, 10-Q), earnings call transcripts, and corporate disclosures.
    Use for: ticker filings, compliance checks, fiscal statements, financial metrics.
    """
    max_results = min(max(1, max_results), 10)
    data = await _async_valyu_query(
        query=query,
        vertical="finance",
        max_num_results=max_results,
        relevance_threshold=relevance_threshold,
    )
    if not data:
        return "Unable to fetch financial context from Valyu."
    return format_valyu_results(data)


@mcp.tool()
async def valyu_context(query: str, max_num_results: int = 5) -> str:
    """
    General web and multi-source context search across all standard Valyu indexes.
    Use as a fallback when queries span multiple disciplines or do not map to academic, biomedical, or finance verticals.
    """
    max_num_results = min(max(1, max_num_results), 10)
    data = await _async_valyu_query(
        query=query,
        vertical=None,
        max_num_results=max_num_results,
    )
    if not data:
        return "Unable to fetch general context from Valyu."
    return format_valyu_results(data)


if __name__ == "__main__":
    VALYU_API_KEY = os.getenv("VALYU_API_KEY")
    if not VALYU_API_KEY:
        logger.error("Error: VALYU_API_KEY environment variable is required")
        sys.exit(1)

    try:
        valyu = Valyu(api_key=VALYU_API_KEY)
        logger.info("Server initialized successfully with Valyu API key")
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Fatal error running MCP server: {str(e)}")
        sys.exit(1)