from typing import Any, Dict, List, Optional
import asyncio
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from valyu import Valyu

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

mcp = FastMCP("valyu-mcp")
valyu: Optional[Valyu] = None


async def _async_valyu_query(
    query: str,
    category: Optional[str] = None,
    max_num_results: int = 5,
    relevance_threshold: float = 0.5,
) -> Optional[Any]:
    """
    Executes a Valyu search inside a background thread via asyncio.to_thread
    using the official category IDs.
    """
    if valyu is None:
        logger.error("Valyu client is not initialized.")
        return None

    try:
        logger.info(
            f"Making Valyu search - query: '{query}', category: {category}, "
            f"max_results: {max_num_results}, threshold: {relevance_threshold}"
        )

        def _call_api():
            kwargs: Dict[str, Any] = {
                "query": query,
                "max_num_results": max_num_results,
                "relevance_threshold": relevance_threshold,
                "search_type": "all",
            }
            if category:
                kwargs["category"] = category

            return valyu.search(**kwargs)

        response = await asyncio.to_thread(_call_api)
        logger.info(f"Received Valyu response with {len(getattr(response, 'results', []) or [])} results")
        return response
    except Exception as e:
        logger.error(f"Error making Valyu request: {str(e)}")
        return None


def format_valyu_results(data: Optional[Any]) -> str:
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
    Uses Valyu 'research' category.
    """
    max_results = min(max(1, max_results), 10)
    data = await _async_valyu_query(
        query=query,
        category="research",
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
    Search life-sciences literature, clinical trials, and health datasets.
    Uses Valyu 'healthcare' category.
    """
    max_results = min(max(1, max_results), 10)
    data = await _async_valyu_query(
        query=query,
        category="healthcare",
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
    Search financial filings, corporate financials, and disclosures.
    Uses Valyu 'company' category.
    """
    max_results = min(max(1, max_results), 10)
    data = await _async_valyu_query(
        query=query,
        category="company",
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
    Fallback when queries do not require domain categorization.
    """
    max_num_results = min(max(1, max_num_results), 10)
    data = await _async_valyu_query(
        query=query,
        category=None,
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