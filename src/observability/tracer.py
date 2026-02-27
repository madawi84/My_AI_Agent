import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger()

@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: dict
    tool_output: str
    duration_ms: float

@dataclass
class AgentStep:
    step_number: int
    reasoning: Optional[str]
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class Trace:
    trace_id: str
    agent_name: str
    input_query: str
    model: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    final_output: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    status: str = "running"
    error: Optional[str] = None

class AgentTracer:
    """
    Captures agent execution flow for debugging and analysis.
    """
    def __init__(self, verbose: bool = False):
        self._traces: dict[str, Trace] = {}
        self._active_trace_id: Optional[str] = None
        self.verbose = verbose

    def start_trace(self, agent_name: str, query: str, model: str = "") -> str:
        """Start a new trace for an agent execution."""
        trace_id = str(uuid.uuid4())[:8]  # Short ID for readability
        self._traces[trace_id] = Trace(
            trace_id=trace_id,
            agent_name=agent_name,
            input_query=query,
            model=model,
        )
        self._active_trace_id = trace_id

        logger.info("trace_started", trace_id=trace_id, agent_name=agent_name, model=model, query=query)
        return trace_id

    def log_step(self, trace_id: str, step: AgentStep):
        """Log a completed step to the trace."""
        if trace_id not in self._traces:
            return

        trace = self._traces[trace_id]
        trace.steps.append(step)

        # Accumulate totals
        trace.total_input_tokens += step.input_tokens
        trace.total_output_tokens += step.output_tokens
        trace.total_cost_usd += step.cost_usd
        trace.total_duration_ms += step.duration_ms

        logger.info("step_completed",
                    trace_id=trace_id,
                    step_number=step.step_number,
                    duration_ms=round(step.duration_ms, 0),
                    cost_usd=round(step.cost_usd, 4))

    def end_trace(self, trace_id: str, output: str, status: str = "completed", error: str = None):
        """Mark a trace as complete."""
        if trace_id not in self._traces:
            return
        trace = self._traces[trace_id]
        trace.final_output = output
        trace.status = status
        trace.error = error

        logger.info("trace_ended",
                    trace_id=trace_id,
                    status=status,
                    duration_ms=round(trace.total_duration_ms, 0),
                    cost_usd=round(trace.total_cost_usd, 4))

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def get_trace_json(self, trace_id: str) -> str:
        """Export a trace as formatted JSON for debugging."""
        if trace_id not in self._traces:
            return "{}"
        return json.dumps(asdict(self._traces[trace_id]), indent=2)

# Global tracer instance
tracer = AgentTracer()
