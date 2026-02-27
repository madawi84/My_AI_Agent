#from __future__ import annotations 

import logging                              
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# use LiteLLM's built-in cost calculator.
try:
    from litellm import completion_cost  
except Exception:  
    completion_cost = None  


@dataclass
class StepCost:
    """
    Cost info for ONE model call (one step in the agent loop).
    """
    step_number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    is_tool_call: bool = False


@dataclass
class QueryCost:
    """
    Stores all steps for one user query and keeps running totals.
    """
    query: str
    steps: list[StepCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: StepCost) -> None:
        """Add a step and update totals."""
        self.steps.append(step)
        self.total_cost_usd += step.cost_usd
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens


class CostTracker:
    """
    Tracks token usage + cost across the full pipeline.
    You call:
      - start_query(query)
      - log_completion(step_number, response)
      - end_query()
    """

    def __init__(self) -> None:
        self.queries: list[QueryCost] = []
        self._current_query: QueryCost | None = None

    def start_query(self, query: str) -> None:
        """Start tracking for a new user query."""
        self._current_query = QueryCost(query=query)

    def end_query(self) -> None:
        """Finish the current query and store it."""
        if self._current_query is not None:
            self.queries.append(self._current_query)
            self._current_query = None

    # Helpers (safe extraction)
    def _get_usage_tokens(self, response: Any) -> tuple[int, int]:
        """
        Extract (input_tokens, output_tokens) safely.

        Providers can return usage in different shapes:
        - response.usage.prompt_tokens / completion_tokens  (Object style)
        - response["usage"]["prompt_tokens"]               (dict style)
        - sometimes input_tokens/output_tokens keys
        """
        usage = None

        # Object style (common in OpenAI/LiteLLM clients)
        if hasattr(response, "usage"):
            usage = getattr(response, "usage", None)

        # Dict style
        if usage is None and isinstance(response, dict):
            usage = response.get("usage") 

        if usage is None:
            return 0, 0

        def pick(obj: Any, key: str) -> int:
            if hasattr(obj, key):
                val = getattr(obj, key)
            elif isinstance(obj, dict):
                val = obj.get(key)
            else:
                val = None
            return int(val) if val is not None else 0

        prompt = pick(usage, "prompt_tokens")
        completion = pick(usage, "completion_tokens")

        # Alternative keys used by some providers
        if prompt == 0:
            prompt = pick(usage, "input_tokens")
        if completion == 0:
            completion = pick(usage, "output_tokens")

        return prompt, completion

    def _get_model_name(self, response: Any) -> str:
        """Extract the model name safely."""
        if hasattr(response, "model"):
            return str(getattr(response, "model") or "")
        if isinstance(response, dict):
            return str(response.get("model") or "")
        return ""

    # Main method
    def log_completion(self, step_number: int, response: Any, is_tool_call: bool = False) -> None:
        """
        Log tokens + cost for one LLM completion.

        - Tokens come from response.usage
        - Cost comes from LiteLLM completion_cost() when available
        """
        if self._current_query is None:
            logger.warning("CostTracker.log_completion called without start_query().")
            return

        # 1) Extract tokens and model name
        input_tokens, output_tokens = self._get_usage_tokens(response)
        model = self._get_model_name(response)

        # 2) Calculate USD cost (best-effort)
        cost_usd = 0.0
        if completion_cost is not None:
            try:
                # LiteLLM calculates cost using its pricing tables
                cost_usd = float(completion_cost(completion_response=response))
            except Exception as e:
                # Not fatal — we still log tokens
                logger.debug(f"completion_cost failed for model='{model}': {e}")
                cost_usd = 0.0

        # 3) Store the step
        step = StepCost(
            step_number=step_number,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            is_tool_call=is_tool_call,
        )
        self._current_query.add_step(step)

    def print_cost_breakdown(self) -> None:
        """
        Print a readable breakdown of the most recent query.
        """
        if not self.queries and self._current_query is None:
            print("No cost data recorded yet.")
            return

        # If the query is still running, show current; else show latest finished.
        q = self._current_query or self.queries[-1]

        print("\n=== COST BREAKDOWN ===")
        print(f"Query: {q.query}")
        print("-" * 72)
        print(f"{'Step':<6}{'Model':<24}{'In':>8}{'Out':>8}{'USD':>10}{'Type':>10}")
        print("-" * 72)

        for s in q.steps:
            step_type = "tool" if s.is_tool_call else "llm"
            print(
                f"{s.step_number:<6}"
                f"{(s.model or '-'): <24}"
                f"{s.input_tokens:>8}"
                f"{s.output_tokens:>8}"
                f"{s.cost_usd:>10.4f}"
                f"{step_type:>10}"
            )

        print("-" * 72)
        print(
            f"{'TOTAL':<30}"
            f"{q.total_input_tokens:>8}"
            f"{q.total_output_tokens:>8}"
            f"{q.total_cost_usd:>10.4f}"
        )
        print()