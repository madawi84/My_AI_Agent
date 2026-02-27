import asyncio
import json
import re
from typing import List, Optional
from dataclasses import dataclass, field

from src.agent.base import BaseAgent
from src.agent.prompts import (
    ANALYST_PROMPT,
    FACT_CHECKER_PROMPT,
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    WRITER_PROMPT,
)
from src.config import settings


@dataclass
class AgentResult:
    """Structured result from any agent."""
    answer: str
    trace_id: str
    total_steps: int = 0


class OrchestratorAgent:
    """
    Planner-first multi-agent pipeline:

      Planner -> (Research subtasks in parallel) -> Analyst -> Fact-check -> Writer

    Returns: {"answer": "...", "metadata": {...}}
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_steps: int = 10,
        default_timeout: float = 30.0,
        research_timeout: float = 60.0,
        max_retries: int = 2,
    ):
        resolved_model = model or settings.model_name
        self.max_steps = max_steps
        self.model = resolved_model
        self.default_timeout = default_timeout
        self.research_timeout = research_timeout
        self.max_retries = max_retries

        # --- Core agents with type hints ---
        self.planner: BaseAgent = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Planner",
            system_prompt=PLANNER_PROMPT,
        )

        self.analyst: BaseAgent = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Analyst",
            system_prompt=ANALYST_PROMPT,
        )

        self.writer: BaseAgent = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Writer",
            system_prompt=WRITER_PROMPT,
        )

        self.fact_checker: BaseAgent = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="FactChecker",
            system_prompt=FACT_CHECKER_PROMPT,
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _get_trace_id(result: dict) -> str:
        """Safely extract trace_id from agent result."""
        return result.get("metadata", {}).get("trace_id", "")

    @staticmethod
    def _parse_result(result: dict) -> AgentResult:
        """Parse agent result into structured AgentResult."""
        return AgentResult(
            answer=result.get("answer", ""),
            trace_id=OrchestratorAgent._get_trace_id(result),
            total_steps=result.get("metadata", {}).get("total_steps", 0),
        )

    # -----------------------------
    # Subtask parsing
    # -----------------------------
    def _extract_subtasks(self, planner_text: str, max_items: int = 5) -> List[str]:
        """
        Convert planner output into a list of subtasks.

        Strategy:
        1. Try JSON first (preferred structured format)
        2. Fall back to regex-based list parsing
        3. Final fallback: treat full text as single task
        """
        if not planner_text:
            return []

        # Strategy 1: Try JSON
        json_tasks = self._try_parse_json(planner_text)
        if json_tasks:
            return json_tasks[:max_items]

        # Strategy 2: Regex-based list parsing
        regex_tasks = self._try_parse_regex(planner_text)
        if regex_tasks:
            return regex_tasks[:max_items]

        # Strategy 3: Fallback - single task
        return [planner_text.strip()]

    def _try_parse_json(self, text: str) -> Optional[List[str]]:
        """Attempt to parse subtasks from JSON output (array or object)."""
        try:
            # Look for JSON in code blocks first
            code_block_match = re.search(
                r'```(?:json)?\s*(\$.*?\$|\{.*?\})\s*```',
                text,
                re.DOTALL,
            )
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Try to find any JSON array or object
                json_match = re.search(r'(\$.*?\$|\{.*?\})', text, re.DOTALL)
                if not json_match:
                    return None
                json_str = json_match.group(1)

            data = json.loads(json_str)

            # Support both JSON array and object formats
            if isinstance(data, list):
                return [item for item in data if isinstance(item, str)]

            if isinstance(data, dict):
                for key in ["tasks", "subtasks", "steps", "items", "plan"]:
                    if key in data and isinstance(data[key], list):
                        return [item for item in data[key] if isinstance(item, str)]

            return None
        except (json.JSONDecodeError, AttributeError):
            return None

    def _try_parse_regex(self, text: str) -> List[str]:
        """Parse subtasks using regex patterns."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        tasks: List[str] = []

        for ln in lines:
            # Match patterns like:
            # "1) task", "2. task", "- task", "* task"
            # "Task 1: task", "Task 1) task", "Step 1: task"
            m = re.match(
                r"^(\d+[\$.\:]|[-*]|Task\s+\d+[:\$]?|Step\s+\d+[:\$]?)\s+(.*)$",
                ln,
                re.IGNORECASE,
            )
            if m:
                task = m.group(2).strip()
                if task:
                    tasks.append(task)

        return tasks

    # -----------------------------
    # Agent execution with timeout + retry
    # -----------------------------
    async def _run_agent_with_retry(
        self,
        agent: BaseAgent,
        query: str,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Run an agent with timeout and optional retry logic.

        Args:
            agent: The agent to run
            query: The query to send
            max_retries: Override instance max_retries (None = use instance default)
            timeout: Override timeout (None = use instance default_timeout)

        Returns:
            Agent result dict
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        timeout = timeout if timeout is not None else self.default_timeout
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(agent.run(query), timeout=timeout)
            except asyncio.TimeoutError:
                last_error = "Timeout"
                if attempt < max_retries:
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    continue

        # All retries exhausted
        return {
            "answer": f"Agent failed after {max_retries + 1} attempts. Last error: {last_error}",
            "metadata": {"trace_id": "", "error": last_error},
        }

    # -----------------------------
    # Parallel researchers
    # -----------------------------
    async def _run_researchers_parallel(
        self,
        subtasks: List[str],
        original_query: str,
    ) -> tuple[List[str], List[str], List[dict]]:
        """
        Run researcher agents in parallel for each subtask.

        Returns:
            Tuple of (research_blocks, trace_ids, raw_results)
        """
        researchers = [
            BaseAgent(
                model=self.model,
                max_steps=self.max_steps,
                agent_name=f"Researcher_{i+1}",
                system_prompt=RESEARCHER_PROMPT,
            )
            for i in range(len(subtasks))
        ]

        async def run_single_research(
            agent: BaseAgent,
            task: str,
            subtask_index: int,
        ) -> tuple[str, str, dict]:
            focused_query = f"Subtask:\n{task}\n\nOriginal user query:\n{original_query}"
            result = await self._run_agent_with_retry(
                agent, focused_query, timeout=self.research_timeout
            )

            # Check if result indicates failure
            error = result.get("metadata", {}).get("error")
            if error:
                return (
                    f"### Researcher_{subtask_index + 1} (FAILED)\nError: {error}",
                    "",
                    result,
                )

            return (
                f"### Researcher_{subtask_index + 1} (Subtask)\n{task}\n\nFindings:\n{result.get('answer', '')}",
                self._get_trace_id(result),
                result,
            )

        # Run all researchers in parallel
        results = await asyncio.gather(
            *(
                run_single_research(agent, task, i)
                for i, (agent, task) in enumerate(zip(researchers, subtasks))
            )
        )

        # Process results
        research_blocks: List[str] = []
        trace_ids: List[str] = []
        raw_results: List[dict] = []

        for block, trace_id, raw in results:
            research_blocks.append(block)
            trace_ids.append(trace_id)
            raw_results.append(raw)

        return research_blocks, trace_ids, raw_results

    # -----------------------------
    # Main orchestration
    # -----------------------------
    async def run(self, query: str) -> dict:
        """
        Orchestration strategy:
        1) Planner: create subtasks
        2) Run researcher agents in parallel for each subtask
        3) Analyst: merge research into structured notes
        4) Fact-checker: verify analyst's brief
        5) Writer: final answer

        Returns {"answer": "...", "metadata": {...}}
        """
        # 1) PLANNER
        planner_result = await self._run_agent_with_retry(
            self.planner, query, timeout=self.default_timeout
        )
        planner_parsed = self._parse_result(planner_result)
        plan_text = planner_parsed.answer
        subtasks = self._extract_subtasks(plan_text, max_items=5)

        if not subtasks:
            subtasks = [query]

        # 2) PARALLEL RESEARCHERS
        (
            research_blocks,
            researcher_trace_ids,
            researcher_results,
        ) = await self._run_researchers_parallel(subtasks, query)
        combined_research = "\n\n".join(research_blocks)

        researcher_steps = sum(
            self._parse_result(r).total_steps for r in researcher_results
        )

        # 3) ANALYST
        analyst_input = (
            f"User query:\n{query}\n\n"
            f"Planner output:\n{plan_text}\n\n"
            f"Parallel research findings:\n{combined_research}\n\n"
            "Task: produce a structured brief with key points, agreements, conflicts, and open questions."
        )
        analyst_result = await self._run_agent_with_retry(
            self.analyst, analyst_input, timeout=45.0
        )
        analyst_parsed = self._parse_result(analyst_result)
        analyst_brief = analyst_parsed.answer

        # 4) FACT-CHECKER
        fact_check_input = (
            f"User query:\n{query}\n\n"
            f"Analyst brief:\n{analyst_brief}\n\n"
            "Task: identify any likely inaccuracies, missing citations, or uncertain claims. "
            "Return a short checklist and a confidence rating (high/medium/low)."
        )
        fact_result = await self._run_agent_with_retry(
            self.fact_checker, fact_check_input, timeout=self.default_timeout
        )
        fact_parsed = self._parse_result(fact_result)
        fact_notes = fact_parsed.answer

        # 5) WRITER
        writer_input = (
            f"User query:\n{query}\n\n"
            f"Analyst brief:\n{analyst_brief}\n\n"
            f"Fact-check notes:\n{fact_notes}\n\n"
            "Task: write the final answer. If something is uncertain, say so clearly."
        )
        writer_result = await self._run_agent_with_retry(
            self.writer, writer_input, timeout=self.default_timeout
        )
        writer_parsed = self._parse_result(writer_result)

        return {
            "answer": writer_parsed.answer,
            "metadata": {
                "planner_trace": planner_parsed.trace_id,
                "researcher_traces": researcher_trace_ids,
                "analyst_trace": analyst_parsed.trace_id,
                "fact_checker_trace": fact_parsed.trace_id,
                "writer_trace": writer_parsed.trace_id,
                "num_subtasks": len(subtasks),
                "total_steps": (
                    planner_parsed.total_steps
                    + researcher_steps
                    + analyst_parsed.total_steps
                    + fact_parsed.total_steps
                    + writer_parsed.total_steps
                ),
            },
        }