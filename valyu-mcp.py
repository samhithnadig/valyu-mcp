try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import Any, Optional, List
import asyncio
from valyu import Valyu, SearchResponse
from mcp.server.fastmcp import FastMCP
import logging

try:
    from valyu.exceptions import (
        ValyuError,
        ValyuAPIError,
        AuthenticationError,
        RateLimitError,
        InvalidRequestError,
    )
except ImportError:
    ValyuError = Exception
    ValyuAPIError = Exception
    AuthenticationError = None
    RateLimitError = None
    InvalidRequestError = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
logger = logging.getLogger(__name__)

mcp = FastMCP("valyu-mcp")
valyu = None

# Verified dataset IDs from Valyu documentation (https://docs.valyu.ai)
ACADEMIC_SOURCES = ["valyu/valyu-arxiv"]

BIOMEDICAL_SOURCES = [
    "valyu/valyu-pubmed",
    "valyu/valyu-biorxiv",
    "valyu/valyu-medrxiv",
    "valyu/valyu-chembl",
    "valyu/valyu-pubchem",
    "valyu/valyu-clinical-trials",
    "valyu/valyu-drug-labels",
]

# Real dataset ID from Valyu documentation (https://docs.valyu.ai/use-cases/finance)
# Note: Marked as unverified if your account/tier does not include the SEC filings subscription.
FINANCE_SOURCES = ["valyu/valyu-sec-filings"]


def format_valyu_error(e: Exception) -> str:
    """Map Valyu SDK and network exceptions to clear, actionable user messages."""
    status_code = getattr(e, "status_code", None)
    err_msg = str(e).strip()

    # 1. Authentication / Invalid API key
    if (
        (AuthenticationError is not None and isinstance(e, AuthenticationError))
        or status_code == 401
        or "401" in err_msg
        or "unauthorized" in err_msg.lower()
        or "invalid api key" in err_msg.lower()
    ):
        return (
            "Error: Invalid or missing Valyu API key (HTTP 401). "
            "Please verify that your VALYU_API_KEY environment variable is set correctly."
        )

    # 2. Insufficient credits / Payment required
    if (
        status_code in (402, 403)
        or "credit" in err_msg.lower()
        or "balance" in err_msg.lower()
        or "payment required" in err_msg.lower()
        or "quota" in err_msg.lower()
    ):
        return (
            f"Error: Insufficient Valyu API credits or access forbidden (HTTP {status_code or 402}). "
            "Please check your account credits at https://valyu.ai."
        )

    # 3. Rate limits
    if (
        (RateLimitError is not None and isinstance(e, RateLimitError))
        or status_code == 429
        or "rate limit" in err_msg.lower()
    ):
        return (
            "Error: Valyu API rate limit reached (HTTP 429). "
            "Please slow down requests or retry shortly."
        )

    # 4. Network / Timeout errors
    if (
        isinstance(e, (TimeoutError, ConnectionError))
        or "timeout" in err_msg.lower()
        or "connection" in err_msg.lower()
        or "timed out" in err_msg.lower()
    ):
        return (
            f"Error: Network timeout or connection failure when reaching Valyu API ({type(e).__name__}): {err_msg}"
        )

    # 5. Invalid request / parameters
    if (
        (InvalidRequestError is not None and isinstance(e, InvalidRequestError))
        or status_code == 400
    ):
        return f"Error: Invalid request to Valyu API (HTTP 400): {err_msg}"

    # 6. Fallback general error
    return f"Error executing request via Valyu API ({type(e).__name__}): {err_msg}"


def _sync_valyu_search(
    client: Any,
    query: str,
    max_num_results: int,
    included_sources: Optional[List[str]] = None,
) -> SearchResponse:
    """
    Synchronous worker executing Valyu search. Prefers modern valyu.search,
    falling back to legacy valyu.context for backward compatibility.
    """
    if hasattr(client, "search"):
        kwargs = {"query": query, "max_num_results": max_num_results}
        if included_sources is not None:
            kwargs["included_sources"] = included_sources
        return client.search(**kwargs)
    elif hasattr(client, "context"):
        return client.context(
            query=query,
            search_type="all",
            max_num_results=max_num_results,
            max_price=10,
            query_rewrite=False,
        )
    else:
        raise AttributeError("Valyu client does not support 'search' or 'context'")


async def make_valyu_request(
    query: str, max_num_results: int = 10, max_price: int = 10
) -> SearchResponse:
    """
    Initiates a request to the Valyu API, incorporating error handling and logging for debugging purposes.
    Offloaded to a worker thread via asyncio.to_thread to prevent blocking the event loop.

    Parameters:
    - query (str): The search query to be sent to the Valyu API.
    - max_num_results (int, optional): The maximum number of results to be returned. Defaults to 10.
    - max_price (int, optional): The maximum price for the results. Defaults to 10.

    Returns:
    - SearchResponse: The response received from the Valyu API.

    Raises:
    - Exception: Re-raises SDK or network exceptions for structured handling.
    """
    if valyu is None:
        raise ValueError("Valyu client is not initialized. Please ensure VALYU_API_KEY is provided.")

    try:
        logger.info(
            f"Making Valyu request - query: {query}, max_results: {max_num_results}, max_price: {max_price}"
        )
        response = await asyncio.to_thread(
            _sync_valyu_search, valyu, query, max_num_results
        )
        logger.info(f"Received Valyu response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error making Valyu request: {str(e)}")
        raise


async def _execute_vertical_search(
    tool_name: str,
    query: str,
    included_sources: List[str],
    max_num_results: int = 5,
) -> str:
    """Helper to execute vertical searches with source filtering and error handling."""
    logger.info(
        f"Starting {tool_name} tool call - query: {query}, max_results: {max_num_results}"
    )
    if valyu is None:
        return "Error: Valyu client is not initialized. Please ensure VALYU_API_KEY is provided."

    try:
        max_num_results = min(max(1, max_num_results), 5)
        data = await asyncio.to_thread(
            _sync_valyu_search,
            valyu,
            query,
            max_num_results,
            included_sources,
        )
        if not data or not data.results:
            return f"No results found for query: '{query}'."

        formatted_results = format_valyu_results(data)
        logger.info(f"Successfully formatted {len(data.results)} results for {tool_name}")
        return formatted_results
    except Exception as e:
        logger.error(f"Error in {tool_name}: {str(e)}")
        return format_valyu_error(e)


def format_valyu_results(data: SearchResponse) -> str:
    """
    Formats the Valyu context results for feeding back to the model as context.

    Parameters:
    - data (SearchResponse): The SearchResponse object received from the Valyu API.

    Returns:
    - str: A formatted string containing the Valyu context results.
    """
    if not data or not data.results:
        return "No results found."

    results = []
    for i, result in enumerate(data.results, 1):
        results.append(
            f"""Title: {result.title}
Content: {result.content or getattr(result, 'description', '')}
URL: {result.url}"""
        )

    formatted_output = "\n\n".join(results)
    return formatted_output


@mcp.tool()
async def valyu_context(query: str, max_num_results: int = 10) -> str:
    """
    A powerful search tool that provides relevant, up-to-date information from across the web and proprietary data sources.
    Use this tool whenever you need to:
    1. Find current information about any topic
    2. Research specific questions or concepts
    3. Get context about companies, technologies, or trends
    4. Verify facts or gather supporting information
    5. Access real-world data and examples

    Parameters:
    - query (str): The search query. Be specific and detailed in your query for best results.
    - max_num_results (int, optional): Maximum number of results to return (default: 10, max: 5)

    Returns:
    - str: Formatted search results including titles, content, and source URLs.
    """
    logger.info(
        f"Starting valyu_context tool call - query: {query}, max_results: {max_num_results}"
    )

    try:
        # Ensure count is within bounds but keep it smaller for faster responses
        max_num_results = min(max(1, max_num_results), 5)  # Reduced from 20 to 5

        data = await make_valyu_request(query, max_num_results, 10)
        if not data or not data.results:
            return f"No results found for query: '{query}'."

        formatted_results = format_valyu_results(data)
        logger.info(f"Successfully formatted {len(data.results)} results")
        return formatted_results

    except Exception as e:
        logger.error(f"Error in valyu_context: {str(e)}")
        return format_valyu_error(e)


@mcp.tool()
async def valyu_academic_search(query: str, max_num_results: int = 5) -> str:
    """
    Search academic and scientific research literature across arXiv.
    Best for finding computer science, physics, mathematics, and AI papers.

    Parameters:
    - query (str): The academic or scientific search query.
    - max_num_results (int, optional): Maximum number of results to return (default: 5, max: 5).

    Returns:
    - str: Formatted search results including paper titles, abstracts/content, and URLs.
    """
    return await _execute_vertical_search(
        "valyu_academic_search", query, ACADEMIC_SOURCES, max_num_results
    )


@mcp.tool()
async def valyu_biomedical_search(query: str, max_num_results: int = 5) -> str:
    """
    Search biomedical, clinical, chemical, and life sciences databases.
    Covers PubMed, bioRxiv, medRxiv, ChEMBL, PubChem, ClinicalTrials.gov, and FDA drug labels.

    Parameters:
    - query (str): The biomedical, clinical, or chemical search query.
    - max_num_results (int, optional): Maximum number of results to return (default: 5, max: 5).

    Returns:
    - str: Formatted search results including study titles, excerpts, and URLs.
    """
    return await _execute_vertical_search(
        "valyu_biomedical_search", query, BIOMEDICAL_SOURCES, max_num_results
    )


@mcp.tool()
async def valyu_finance_search(query: str, max_num_results: int = 5) -> str:
    """
    Search corporate financial filings, 10-K/10-Q reports, and disclosures.
    Best for analyzing public company financials, risk factors, and regulatory disclosures.

    Parameters:
    - query (str): The financial or company disclosure search query.
    - max_num_results (int, optional): Maximum number of results to return (default: 5, max: 5).

    Returns:
    - str: Formatted search results including filing titles, excerpts, and URLs.
    """
    return await _execute_vertical_search(
        "valyu_finance_search", query, FINANCE_SOURCES, max_num_results
    )


@mcp.tool()
async def valyu_answer(query: str) -> str:
    """
    Get a direct, synthesized answer backed by cited sources from Valyu.
    Best for answering questions that need a clear, factually grounded response with references.

    Parameters:
    - query (str): The question to answer.

    Returns:
    - str: Synthesized answer text and citation links.
    """
    logger.info(f"Starting valyu_answer tool call - query: {query}")
    if valyu is None:
        return "Error: Valyu client is not initialized. Please ensure VALYU_API_KEY is provided."

    try:
        if not hasattr(valyu, "answer"):
            return "Error: Installed Valyu SDK does not support 'valyu.answer'. Please upgrade the valyu package."

        response = await asyncio.to_thread(valyu.answer, query=query)
        if not response:
            return "No answer returned by Valyu API."

        # Handle field differences across SDK versions (answer vs contents)
        answer_text = (
            getattr(response, "contents", None)
            or getattr(response, "answer", None)
            or (response.get("contents") or response.get("answer") if isinstance(response, dict) else None)
            or str(response)
        )

        sources = (
            getattr(response, "sources", None)
            or getattr(response, "citations", None)
            or (response.get("sources") or response.get("citations") if isinstance(response, dict) else None)
            or []
        )

        formatted_sources = []
        for s in sources:
            title = getattr(s, "title", None) or (s.get("title") if isinstance(s, dict) else "Source")
            url = getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else "")
            if url:
                formatted_sources.append(f"- [{title}]({url})")
            else:
                formatted_sources.append(f"- {title}")

        if formatted_sources:
            return f"{answer_text}\n\n**Sources:**\n" + "\n".join(formatted_sources)
        return str(answer_text)

    except Exception as e:
        logger.error(f"Error in valyu_answer: {str(e)}")
        return format_valyu_error(e)


@mcp.tool()
async def valyu_start_deep_research(query: str, mode: str = "standard") -> str:
    """
    Start an asynchronous autonomous deep research task with Valyu DeepResearch.
    Returns a task ID that can be used with valyu_get_research_status to check progress.

    Parameters:
    - query (str): The research question or topic.
    - mode (str, optional): Research mode ('fast', 'standard', 'heavy', 'max'). Default: 'standard'.

    Returns:
    - str: Confirmation message containing the task_id to poll.
    """
    logger.info(f"Starting valyu_start_deep_research - query: {query}, mode: {mode}")
    if valyu is None:
        return "Error: Valyu client is not initialized. Please ensure VALYU_API_KEY is provided."

    try:
        if not hasattr(valyu, "deepresearch"):
            return "Error: Installed Valyu SDK does not support 'valyu.deepresearch'. Please upgrade the valyu package."

        task = await asyncio.to_thread(valyu.deepresearch.create, query=query, mode=mode)
        task_id = (
            getattr(task, "deepresearch_id", None)
            or getattr(task, "task_id", None)
            or getattr(task, "id", None)
            or (task.get("deepresearch_id") or task.get("task_id") or task.get("id") if isinstance(task, dict) else None)
        )
        if not task_id:
            return f"Deep research started, but could not determine task ID from response: {task}"

        return (
            f"Deep research task started successfully.\n"
            f"- **Task ID:** `{task_id}`\n"
            f"- **Mode:** {mode}\n\n"
            f"Use `valyu_get_research_status(task_id='{task_id}')` to monitor progress or retrieve the report."
        )
    except Exception as e:
        logger.error(f"Error in valyu_start_deep_research: {str(e)}")
        return format_valyu_error(e)


@mcp.tool()
async def valyu_get_research_status(task_id: str) -> str:
    """
    Check the status or retrieve results of an existing Valyu DeepResearch task.

    Parameters:
    - task_id (str): The ID returned by valyu_start_deep_research.

    Returns:
    - str: Current status, research report (if completed), or error message (if failed).
    """
    logger.info(f"Checking deep research status for task_id: {task_id}")
    if valyu is None:
        return "Error: Valyu client is not initialized. Please ensure VALYU_API_KEY is provided."

    try:
        if not hasattr(valyu, "deepresearch"):
            return "Error: Installed Valyu SDK does not support 'valyu.deepresearch'. Please upgrade the valyu package."

        status_obj = await asyncio.to_thread(valyu.deepresearch.status, task_id)
        status_val = (
            getattr(status_obj, "status", None)
            or (status_obj.get("status") if isinstance(status_obj, dict) else None)
            or "unknown"
        )

        if status_val == "completed":
            report = (
                getattr(status_obj, "output", None)
                or getattr(status_obj, "report", None)
                or (status_obj.get("output") or status_obj.get("report") if isinstance(status_obj, dict) else "")
            )
            pdf_url = (
                getattr(status_obj, "pdf_url", None)
                or (status_obj.get("pdf_url") if isinstance(status_obj, dict) else None)
            )
            cost = (
                getattr(status_obj, "cost", None)
                or (status_obj.get("cost") if isinstance(status_obj, dict) else None)
            )
            out = f"### DeepResearch Task Completed ({task_id})\n\n{report}"
            if pdf_url:
                out += f"\n\n**PDF Report:** [Download]({pdf_url})"
            if cost is not None:
                out += f"\n**Cost:** ${cost}"
            return out
        elif status_val in ("queued", "running", "processing", "pending"):
            return (
                f"Task `{task_id}` is currently **{status_val}**.\n"
                "Deep research runs asynchronously and may take several minutes depending on depth. "
                "Please poll again in a few moments."
            )
        elif status_val == "failed":
            err = (
                getattr(status_obj, "error", None)
                or (status_obj.get("error") if isinstance(status_obj, dict) else "Unknown error")
            )
            return f"Task `{task_id}` failed: {err}"
        else:
            return f"Task `{task_id}` status: {status_val}"

    except Exception as e:
        logger.error(f"Error checking research status for {task_id}: {str(e)}")
        return format_valyu_error(e)


if __name__ == "__main__":
    import os

    VALYU_API_KEY = os.getenv("VALYU_API_KEY")
    if not VALYU_API_KEY:
        logger.error("Error: VALYU_API_KEY environment variable is required")
        exit(1)

    try:
        valyu = Valyu(api_key=VALYU_API_KEY)
        logger.info("Server initialized successfully with Valyu API key")

        # Initialize and run the server using stdio transport
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Fatal error running MCP server: {str(e)}")
        exit(1)
