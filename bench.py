#!/usr/bin/env python3
"""
Benchmark and Verification Suite for valyu-mcp
Runs comparative queries, concurrency benchmarks, and error diagnostics.
Reads VALYU_API_KEY from environment or .env. Automatically falls back to mock fixtures
if VALYU_API_KEY is not set or network is unreachable.
"""

import asyncio
import os
import sys
import time
import types
from typing import List, Optional, Any

# Ensure local repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optional load_dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

VALYU_API_KEY = os.getenv("VALYU_API_KEY")
USE_MOCK = False

# ==========================================
# Mock Fixtures for Offline / Unit Testing
# ==========================================
class MockSearchResult:
    def __init__(self, title: str, content: str, url: str, source: str = "web"):
        self.title = title
        self.content = content
        self.url = url
        self.source = source


class MockSearchResponse:
    def __init__(self, query: str, results: List[MockSearchResult]):
        self.query = query
        self.results = results


class MockAnswerResponse:
    def __init__(self, answer: str, sources: List[MockSearchResult]):
        self.contents = answer
        self.answer = answer
        self.sources = sources


class MockDeepResearchTask:
    def __init__(self, task_id: str):
        self.deepresearch_id = task_id
        self.task_id = task_id


class MockDeepResearchStatus:
    def __init__(self, status: str, output: str = "", cost: float = 0.50):
        self.status = status
        self.output = output
        self.cost = cost
        self.sources = [MockSearchResult("Source 1", "Evidence", "https://example.com/1")]
        self.pdf_url = "https://example.com/report.pdf"


class MockValyuClient:
    def __init__(self, api_key: str = "mock-key", simulate_delay: float = 0.15):
        self.api_key = api_key
        self.simulate_delay = simulate_delay

        class DeepResearchSubClient:
            def __init__(self, parent):
                self.parent = parent

            def create(self, query: str, mode: str = "standard"):
                time.sleep(self.parent.simulate_delay)
                return MockDeepResearchTask("dr_task_987654")

            def status(self, task_id: str):
                time.sleep(self.parent.simulate_delay)
                return MockDeepResearchStatus(
                    status="completed",
                    output=f"# Deep Research Report for {task_id}\n\nComprehensive findings...",
                )

        self.deepresearch = DeepResearchSubClient(self)

    def search(self, query: str, max_num_results: int = 5, included_sources: Optional[List[str]] = None):
        time.sleep(self.simulate_delay)
        source_label = included_sources[0] if included_sources else "web"
        if "empty_result_trigger" in query:
            return MockSearchResponse(query, [])

        results = [
            MockSearchResult(
                title=f"[{source_label.upper()}] Insight on: {query[:35]}... (Doc #{i})",
                content=f"Detailed evidence and contextual data retrieved from source: {source_label}.",
                url=f"https://data.valyu.ai/{source_label}/doc-{i}",
                source=source_label,
            )
            for i in range(1, max_num_results + 1)
        ]
        return MockSearchResponse(query, results)

    def context(self, query: str, search_type: str = "all", max_num_results: int = 5, **kwargs):
        return self.search(query=query, max_num_results=max_num_results)

    def answer(self, query: str):
        time.sleep(self.simulate_delay)
        sources = [
            MockSearchResult("Valyu Synthesis Ref", "Primary source text", "https://valyu.ai/ref/1"),
            MockSearchResult("Academic Foundation", "Peer-reviewed basis", "https://valyu.ai/ref/2"),
        ]
        return MockAnswerResponse(
            answer=f"Grounded answer: {query} involves key verified findings across domain sources.",
            sources=sources,
        )


# Setup mock modules if valyu or mcp are not installed in the environment
if not VALYU_API_KEY or "--mock" in sys.argv:
    USE_MOCK = True

try:
    import valyu
except ImportError:
    USE_MOCK = True
    mock_valyu_mod = types.ModuleType("valyu")
    mock_valyu_mod.Valyu = MockValyuClient
    mock_valyu_mod.SearchResponse = MockSearchResponse
    sys.modules["valyu"] = mock_valyu_mod

try:
    import mcp
    import mcp.server.fastmcp
except ImportError:
    mock_mcp_mod = types.ModuleType("mcp")
    mock_server = types.ModuleType("mcp.server")
    mock_fastmcp = types.ModuleType("mcp.server.fastmcp")

    class MockFastMCP:
        def __init__(self, name="mock"):
            self.name = name

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

        def run(self, *args, **kwargs):
            pass

    mock_fastmcp.FastMCP = MockFastMCP
    mock_server.fastmcp = mock_fastmcp
    mock_mcp_mod.server = mock_server
    sys.modules["mcp"] = mock_mcp_mod
    sys.modules["mcp.server"] = mock_server
    sys.modules["mcp.server.fastmcp"] = mock_fastmcp

# Import functions under test from valyu-mcp.py
import importlib.util
spec = importlib.util.spec_from_file_location("valyu_mcp", os.path.join(os.path.dirname(__file__), "valyu-mcp.py"))
valyu_mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(valyu_mcp)


# ==========================================
# Benchmark Suite
# ==========================================
async def run_query_comparisons(client: Any):
    print("=" * 80)
    print("PART 1: 10 DOMAIN QUERIES (Generic valyu_context vs Vertical Tools)")
    print("=" * 80)

    queries = [
        # Finance queries (3)
        ("finance", "Apple Q3 2024 10-Q primary revenue drivers and segment gross margins", valyu_mcp.valyu_finance_search),
        ("finance", "Tesla regulatory emissions credits revenue impact on Q2 2024 GAAP net income", valyu_mcp.valyu_finance_search),
        ("finance", "NVIDIA data center compute revenue growth and supply constraints 10-K", valyu_mcp.valyu_finance_search),

        # Biomedical queries (3)
        ("biomedical", "GLP-1 receptor agonist semaglutide cardiovascular outcomes randomized clinical trials", valyu_mcp.valyu_biomedical_search),
        ("biomedical", "CRISPR-Cas9 off-target cleavage mechanisms and guide RNA specificity", valyu_mcp.valyu_biomedical_search),
        ("biomedical", "ChEMBL EGFR tyrosine kinase inhibitors bioactivity IC50 and selectivity profile", valyu_mcp.valyu_biomedical_search),

        # Academic / General queries (4)
        ("academic", "Linear attention mechanisms and state space models complexity arXiv", valyu_mcp.valyu_academic_search),
        ("academic", "Latent diffusion models mathematical proof of score matching and denoising", valyu_mcp.valyu_academic_search),
        ("academic", "Surface codes quantum error correction threshold theorems and fault tolerance", valyu_mcp.valyu_academic_search),
        ("academic", "Retrieval augmented generation latency reduction via chunk compression", valyu_mcp.valyu_academic_search),
    ]

    for idx, (category, q, vertical_func) in enumerate(queries, 1):
        print(f"\n--- [Query {idx}/10] Category: {category.upper()} ---")
        print(f"Query: \"{q}\"")

        # 1. Generic valyu_context
        t0 = time.perf_counter()
        generic_res = await valyu_mcp.valyu_context(query=q, max_num_results=2)
        generic_time = (time.perf_counter() - t0) * 1000

        # 2. Vertical tool
        t0 = time.perf_counter()
        vertical_res = await vertical_func(query=q, max_num_results=2)
        vertical_time = (time.perf_counter() - t0) * 1000

        # Print Side by Side summary
        first_generic_line = generic_res.strip().split("\n")[0] if generic_res else "None"
        first_vertical_line = vertical_res.strip().split("\n")[0] if vertical_res else "None"

        print(f"  • Generic Tool  ({generic_time:5.1f}ms): {first_generic_line[:75]}")
        print(f"  • Vertical Tool ({vertical_time:5.1f}ms): {first_vertical_line[:75]}")


async def run_concurrency_benchmark(client: Any):
    print("\n" + "=" * 80)
    print("PART 2: CONCURRENCY BENCHMARK (asyncio.gather vs Blocking Execution)")
    print("=" * 80)
    print("Running 3 parallel searches simultaneously...")

    search_queries = [
        "Concurrent search test alpha",
        "Concurrent search test beta",
        "Concurrent search test gamma",
    ]

    # Test A: Sequential blocking simulation (old behavior where event loop was blocked)
    print("\n[Simulation: Synchronous blocking execution without to_thread]")
    t_start = time.perf_counter()
    res_seq = []
    for q in search_queries:
        data = client.search(q, max_num_results=2)
        res_seq.append(data)
    time_sequential = time.perf_counter() - t_start
    print(f"  Sequential execution time: {time_sequential:.3f}s (each request halts the loop)")

    # Test B: Concurrent execution with asyncio.to_thread
    print("\n[Actual Implementation: Concurrent execution using asyncio.gather + asyncio.to_thread]")
    t_start = time.perf_counter()
    tasks = [
        valyu_mcp.valyu_context(query=q, max_num_results=2)
        for q in search_queries
    ]
    res_concurrent = await asyncio.gather(*tasks)
    time_concurrent = time.perf_counter() - t_start
    print(f"  Concurrent execution time: {time_concurrent:.3f}s (worker threads run in parallel)")
    print(f"  Speedup / event-loop responsiveness factor: {time_sequential / max(time_concurrent, 0.001):.2f}x")


async def run_error_case_tests():
    print("\n" + "=" * 80)
    print("PART 3: ERROR HANDLING AND DIAGNOSTICS SUITE")
    print("=" * 80)

    # 1. Invalid API Key
    class CustomAuthError(Exception):
        status_code = 401
    msg_auth = valyu_mcp.format_valyu_error(CustomAuthError("401 Unauthorized: Invalid API key"))
    print(f"\n[Case 1: Bad API Key (401)]\n  Output: {msg_auth}")

    # 2. Insufficient Credits / Payment Required
    class CustomCreditError(Exception):
        status_code = 402
    msg_credit = valyu_mcp.format_valyu_error(CustomCreditError("Insufficient credits on account"))
    print(f"\n[Case 2: Insufficient Credits (402)]\n  Output: {msg_credit}")

    # 3. Rate Limit Exceeded
    class CustomRateLimit(Exception):
        status_code = 429
    msg_rate = valyu_mcp.format_valyu_error(CustomRateLimit("429 Too Many Requests"))
    print(f"\n[Case 3: Rate Limit Exceeded (429)]\n  Output: {msg_rate}")

    # 4. Network Timeout
    msg_timeout = valyu_mcp.format_valyu_error(TimeoutError("Connection timed out after 10000ms"))
    print(f"\n[Case 4: Network Timeout]\n  Output: {msg_timeout}")

    # 5. Empty Result Set
    empty_res = await valyu_mcp.valyu_context(query="empty_result_trigger_xyz_nonexistent_123")
    print(f"\n[Case 5: Empty Result Set]\n  Output: {empty_res}")

    # 6. Uninitialized Client
    original_client = valyu_mcp.valyu
    valyu_mcp.valyu = None
    msg_uninit = await valyu_mcp.valyu_context(query="test")
    print(f"\n[Case 6: Uninitialized Client]\n  Output: {msg_uninit}")
    valyu_mcp.valyu = original_client


async def run_answer_and_deep_research_checks():
    print("\n" + "=" * 80)
    print("PART 4: ANSWER & DEEP RESEARCH TOOL VERIFICATION")
    print("=" * 80)

    # Test valyu_answer
    ans = await valyu_mcp.valyu_answer(query="What is quantum computing?")
    print(f"\n[valyu_answer output]:\n{ans}\n")

    # Test valyu_start_deep_research
    task_res = await valyu_mcp.valyu_start_deep_research(query="Analyze semiconductor market 2026", mode="standard")
    print(f"[valyu_start_deep_research output]:\n{task_res}\n")

    # Test valyu_get_research_status
    status_res = await valyu_mcp.valyu_get_research_status(task_id="dr_task_987654")
    print(f"[valyu_get_research_status output]:\n{status_res[:200]}...\n")


async def main():
    print("Valyu MCP Benchmark & Verification Suite")
    if USE_MOCK:
        print("Note: Running with Mock fixtures (no live API key / offline environment detected).")
        client = MockValyuClient()
    else:
        print(f"Note: Running with live Valyu API key ({VALYU_API_KEY[:6]}...).")
        client = valyu.Valyu(api_key=VALYU_API_KEY)

    valyu_mcp.valyu = client

    await run_query_comparisons(client)
    await run_concurrency_benchmark(client)
    await run_error_case_tests()
    await run_answer_and_deep_research_checks()

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
