from __future__ import annotations

from typing import Any, Dict, List, Optional

from tavily import AsyncTavilyClient


class TavilySearchTool:
    """
    Tavily-backed web search tool.

    Why this exists:
    - DuckDuckGo HTML often blocks automated requests (403).
    - Tavily provides stable, LLM-friendly search results in JSON.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("TAVILY_API_KEY is missing. Add it to your .env")
        self.client = AsyncTavilyClient(api_key)

    async def search_web(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",  # "basic" or "advanced" (per Tavily docs)
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform a Tavily search and normalise the output.

        Returns a dict shaped like:
        {
          "query": "...",
          "results": [{"title":..., "url":..., "content":...}, ...],
          "answer": "... (optional)"
        }
        """
        # Guardrails (avoid provider errors / huge payloads)
        max_results = max(1, min(int(max_results), 10))

        # Tavily docs: client.search(query=..., ...) and parameters table
        resp = await self.client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )

        # Normalise results to a minimal stable schema your agent can rely on
        results: List[Dict[str, Any]] = []
        for r in resp.get("results", []) or []:
            results.append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                    # keep extras if you want:
                    "score": r.get("score"),
                }
            )

        out: Dict[str, Any] = {"query": query, "results": results}

        # If you enable include_answer, Tavily may return an "answer"
        if "answer" in resp:
            out["answer"] = resp.get("answer")

        return out