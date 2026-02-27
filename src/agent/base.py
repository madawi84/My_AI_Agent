"""
BaseAgent: a ReAct agent with built-in observability.

Complete sections are marked COMPLETE.
The core run() method is marked TODO — that is your main implementation challenge.
"""

import asyncio
import json
from pyexpat.errors import messages
import time
from typing import Optional

import structlog
from litellm import acompletion, completion_cost
from pydantic import ValidationError

from src.agent.prompts import DEFAULT_SYSTEM_PROMPT
from src.config import settings
from src.observability.loop_detector import AdvancedLoopDetector
from src.observability.tracer import AgentStep, AgentTracer, ToolCallRecord
from src.tools.registry import registry

logger = structlog.get_logger()


class BaseAgent:
    """
    A ReAct agent with full observability:
    - Structured tracing of every step
    - Loop detection (exact, fuzzy, stagnation)
    - Per-step cost tracking
    - Async execution with parallel tool calling
    """

    # ── COMPLETE: __init__ ────────────────────────────────────────────────

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 10,
        agent_name: str = "BaseAgent",
        verbose: bool = True,
        system_prompt: str | None = None,
        tools: list | None = None,
    ):
        self.model = model or settings.model_name
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if tools is None:
            self.tools = registry.get_all_tools()
        else:
            self.tools = tools

        self.tools_schema = [tool.to_openai_schema() for tool in self.tools]

        # Observability stack
        self.tracer = AgentTracer(verbose=verbose)
        self.loop_detector = AdvancedLoopDetector()

        # Shared state across hooks within a single run()
        self._current_trace_id: Optional[str] = None

    # ── COMPLETE: Tool execution ───────────────────────────────────────────

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Registry lookup + loop detection + asyncio.to_thread + error handling."""
        # Check for loops BEFORE executing
        loop_check = self.loop_detector.check_tool_call(
            tool_name, json.dumps(arguments)
        )
        if loop_check.is_looping:
            logger.warning(
                "loop_detected",
                tool=tool_name,
                strategy=loop_check.strategy,
                message=loop_check.message,
            )
            return (
                f"SYSTEM: {loop_check.message} "
                f"(Detection: {loop_check.strategy})"
            )

        tool = registry.get_tool(tool_name)
        if not tool:
            logger.error("tool_not_found", tool=tool_name)
            return f"Error: Tool '{tool_name}' not found."
        try:
            result = await asyncio.to_thread(tool.execute, **arguments)
            return str(result)
        except ValidationError as e:
            logger.warning("tool_validation_failed", tool=tool_name, error=str(e))
            return f"Error: Tool arguments validation failed. {e}"
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            return f"Error: {type(e).__name__}: {e}"

    # ── COMPLETE: Hooks ────────────────────────────────────────────────────

    def _on_step_start(self, step: int, messages: list) -> None:
        """Called at the start of each ReAct step. Override to add custom behaviour."""
        pass

    def _on_tool_result(
        self, step: int, name: str, args: dict, result: str, duration_ms: float
    ) -> None:
        """Called after each tool result is received. Override to add custom behaviour."""
        pass

    def _on_step_end(
        self, step: int, response, tool_calls: list, step_duration_ms: float
    ) -> None:
        """
        Log a completed step to the tracer and record per-step cost.

        tool_calls is a list of (name, args, result, duration_ms) tuples
        accumulated during this step.
        """
        message = response.choices[0].message

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        try:
            step_cost = completion_cost(completion_response=response)
        except Exception:
            step_cost = 0.0

        tool_records = [
            ToolCallRecord(
                tool_name=name,
                tool_input=args,
                tool_output=result,
                duration_ms=dur_ms,
            )
            for name, args, result, dur_ms in tool_calls
        ]

        agent_step = AgentStep(
            step_number=step,
            reasoning=message.content,
            tool_calls=tool_records,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=step_cost,
            duration_ms=step_duration_ms,
        )
        self.tracer.log_step(self._current_trace_id, agent_step)

    def _on_loop_end(
        self,
        answer: str,
        total_steps: int,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """Finalise the trace when the ReAct loop exits (normally or on error)."""
        end_kwargs = {}
        if status != "success":
            end_kwargs["status"] = status
        if error:
            end_kwargs["error"] = error
        self.tracer.end_trace(self._current_trace_id, answer, **end_kwargs)

    # ── TODO: run() ───────────────────────────────────────────────────────

    async def run(self, user_query: str) -> dict:
        """
        Execute the ReAct (Reasoning + Acting) loop to answer a user query.
        The agent iteratively:
        1) Calls the LLM with the current conversation state.
        2) If the LLM requests tool calls, execute them (in parallel) and append results.
        3) Repeat until the LLM returns a final answer or max_steps is reached.

        Returns:
            dict: {
                "answer": str,
                "metadata": {"trace_id": str, "total_steps": int}
            }
        """
        # 0) Initialise trace + messages
        self._current_trace_id = self.tracer.start_trace(
            agent_name=self.agent_name,
            model=self.model,
            query=user_query,
        )

        # Conversation state (OpenAI/LiteLLM chat format)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        # store the “best” final answer here
        final_answer: str = ""
        status: str = "success"
        error_msg: Optional[str] = None

        total_steps_completed = 0

        try:
            # 1) Main ReAct loop
            for step in range(1, self.max_steps + 1):
                total_steps_completed = step

                # Hook: step start (lets you add custom logging/printing later)
                self._on_step_start(step, messages)

                step_start = time.time()

                # 2) Call the LLM (one completion per step)
                try:
                    response = await acompletion(
                        model=self.model,
                        messages=messages,
                        tools=self.tools_schema,
                        tool_choice="auto",
                        max_tokens=getattr(settings, "max_tokens", 256),
                        temperature=0.2,
                    )
                except Exception as e:
                    # Graceful failure: end loop with error
                    status = "error"
                    error_msg = f"LLM call failed: {type(e).__name__}: {e}"
                    final_answer = error_msg #"Sorry — I couldn’t complete the request due to an LLM error."
                    break

                message = response.choices[0].message
                tool_calls = message.tool_calls or [] # Extract tool calls from the LLM response (if any)

                # Record of tool calls for this step (for tracer)
                step_tool_records: list[tuple[str, dict, str, float]] = []

                # 3) If there are NO tool calls-> final answer
                if not tool_calls:
                    final_answer = message.content or ""
                    step_duration_ms = (time.time() - step_start) * 1000.0

                    # Hook: step end (logs to tracer + cost via completion_cost already in hook)
                    self._on_step_end(step, response, step_tool_records, step_duration_ms)

                    # Normal exit
                    break

                # If there are tool calls, record the assistant message
                # that triggered them (helps the model keep context).
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        # Important: include tool_calls so the next messages can reference them
                        "tool_calls": tool_calls,
                    }
                )

                # 4) Execute ALL tool calls in parallel
                async def run_one_tool(call) -> tuple[str, dict, str, float]:
                    """
                    Execute a single tool call and return:
                    (tool_name, args, result_str, duration_ms)
                    """
                    tool_name = call.function.name
                    raw_args = call.function.arguments or "{}"

                    # Parse tool arguments safely
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            args = {}
                    except Exception:
                        args = {}

                    tool_start = time.time()
                    result_str = await self._execute_tool(tool_name, args)
                    dur_ms = (time.time() - tool_start) * 1000.0
                    return tool_name, args, result_str, dur_ms

                # Run all tools at once (parallel)
                results = await asyncio.gather(
                    *(run_one_tool(c) for c in tool_calls),
                    return_exceptions=True,
                )

                # 5) Append tool results back into messages
                for idx, item in enumerate(results):
                    call = tool_calls[idx]   # ✅ store the matching tool call

                    if isinstance(item, Exception):
                        tool_name = call.function.name
                        result_str = f"Tool execution failed: {item}"
                    else:
                        tool_name, result = item
                        result_str = result if isinstance(result, str) else json.dumps(result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[idx].id,  # use stored call
                            "name": tool_name,
                            "content": result_str,
                        }
                    )
                step_duration_ms = (time.time() - step_start) * 1000.0

                # Hook: step end (logs trace + cost)
                self._on_step_end(step, response, step_tool_records, step_duration_ms)

            else:
                # reach max_steps without breaking, we return a fallback
                status = "max_steps"
                final_answer = (
                 "I could not finish within the allowed number of steps. "
                 "Try a simpler query or increase max_steps."
                )

        except Exception as e:
            # Catch any unexpected crash and end trace gracefully
            status = "error"
            error_msg = f"Unexpected error: {type(e).__name__}: {e}"
            final_answer = "Sorry — something went wrong while running the agent."

        # 6) Final hook: loop end
        self._on_loop_end(
            answer=final_answer,
            total_steps=total_steps_completed,
            status=status,
            error=error_msg,
        )

        return {
            "answer": final_answer,
            "metadata": {
                "trace_id": self._current_trace_id,
                "total_steps": total_steps_completed,
            },
        }