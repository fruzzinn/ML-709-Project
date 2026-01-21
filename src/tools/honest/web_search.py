"""Web search tool for retrieving information from the web."""

from __future__ import annotations

from typing import Any

import aiohttp

from src.tools.base import BaseTool, ParameterType, ToolExecutionContext, ToolParameter


class WebSearchTool(BaseTool):
    """A web search tool that retrieves search results.

    In a production environment, this would integrate with a real search API
    (e.g., Brave Search, SerpAPI, Bing Search API). For research purposes,
    this implementation provides a mock interface that can be configured
    to return controlled responses for experiments.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        mock_mode: bool = True,
    ) -> None:
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.mock_mode = mock_mode
        self._mock_results: dict[str, list[dict[str, str]]] = {}

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Returns a list of search results "
            "with titles, snippets, and URLs. Useful for finding current information, "
            "facts, documentation, and references."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type=ParameterType.STRING,
                description="The search query to look up",
                required=True,
            ),
            ToolParameter(
                name="num_results",
                type=ParameterType.INTEGER,
                description="Number of results to return (default: 5, max: 10)",
                required=False,
                default=5,
            ),
        ]

    def set_mock_results(self, query: str, results: list[dict[str, str]]) -> None:
        """Set mock results for a specific query (for testing)."""
        self._mock_results[query.lower()] = results

    async def execute(
        self,
        arguments: dict[str, Any],
        _context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Execute the web search."""
        query = arguments.get("query", "")
        num_results = min(arguments.get("num_results", 5), 10)

        if not query:
            return {"error": "No query provided", "results": []}

        if self.mock_mode:
            return await self._mock_search(query, num_results)
        else:
            return await self._real_search(query, num_results)

    async def _mock_search(
        self,
        query: str,
        num_results: int,
    ) -> dict[str, Any]:
        """Return mock search results for testing."""
        # Check for pre-configured mock results
        query_lower = query.lower()
        if query_lower in self._mock_results:
            results = self._mock_results[query_lower][:num_results]
            return {
                "query": query,
                "results": results,
                "num_results": len(results),
            }

        # Generate generic mock results
        results = []
        for i in range(num_results):
            results.append(
                {
                    "title": f"Result {i + 1} for: {query}",
                    "snippet": f"This is a mock search result snippet for the query '{query}'. "
                    f"It contains relevant information about the topic.",
                    "url": f"https://example.com/result-{i + 1}",
                }
            )

        return {
            "query": query,
            "results": results,
            "num_results": len(results),
        }

    async def _real_search(
        self,
        query: str,
        num_results: int,
    ) -> dict[str, Any]:
        """Perform a real web search using configured API."""
        if not self.api_url:
            return {
                "error": "Search API not configured",
                "query": query,
                "results": [],
            }

        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                params = {
                    "q": query,
                    "count": num_results,
                }

                async with session.get(
                    self.api_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        return {
                            "error": f"Search API returned status {response.status}",
                            "query": query,
                            "results": [],
                        }

                    data = await response.json()

                    # Parse results (format depends on the API)
                    results = self._parse_search_response(data)

                    return {
                        "query": query,
                        "results": results[:num_results],
                        "num_results": len(results),
                    }

        except TimeoutError:
            return {
                "error": "Search request timed out",
                "query": query,
                "results": [],
            }
        except Exception as e:
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": [],
            }

    def _parse_search_response(self, data: dict[str, Any]) -> list[dict[str, str]]:
        """Parse search API response into standard format."""
        results = []

        # Handle common API response formats
        items = data.get("results", data.get("items", data.get("webPages", {}).get("value", [])))

        for item in items:
            result = {
                "title": item.get("title", item.get("name", "")),
                "snippet": item.get("snippet", item.get("description", "")),
                "url": item.get("url", item.get("link", "")),
            }
            if result["title"] and result["url"]:
                results.append(result)

        return results
