"""
src/tools/search_tool.py

Web search tool using Tavily API.
This replaces the DuckDuckGo scraper which often causes 403 errors.
"""

import os
import requests
import structlog

from src.tools.registry import registry

logger = structlog.get_logger()

TAVILY_API_URL = "https://api.tavily.com/search"


@registry.register(
    name="search_web",
    description="Search the web for recent information and return summarized results.",
    category="research",
)
def search_web(query: str, max_results: int = 5):
    """
    Search the web using Tavily.

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        Maximum number of results to return.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set in environment")

    try:
        response = requests.post(
            TAVILY_API_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                }
            )

        return {
            "query": query,
            "answer": data.get("answer"),
            "results": results,
        }

    except Exception as e:
        logger.error("tavily_search_failed", error=str(e))
        return {"error": str(e)}