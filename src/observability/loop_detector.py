from dataclasses import dataclass

@dataclass
class LoopDetectionResult:
    is_looping: bool
    strategy: str  # "exact", "fuzzy", "stagnation", "none"
    message: str
    confidence: float

class AdvancedLoopDetector:
    """
    Detects agent loops using three strategies.
    """
    def __init__(
        self,
        exact_threshold: int = 2,
        fuzzy_threshold: float = 0.8,
        stagnation_window: int = 3,
    ):
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.stagnation_window = stagnation_window
        self.tool_history: list[tuple[str, str]] = []  # (tool_name, args_str)
        self.output_history: list[str] = []

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """
        Compute Jaccard similarity between two strings.
        Uses word-level tokens for meaningful comparison.
        """
        # Split on whitespace and ignore purely numeric tokens (e.g. IDs)
        tokens1 = {t for t in s1.lower().split() if not t.isnumeric()}
        tokens2 = {t for t in s2.lower().split() if not t.isnumeric()}

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def check_tool_call(self, tool_name: str, tool_input: str) -> LoopDetectionResult:
        """
        Check if a tool call indicates a loop.
        Call this BEFORE executing the tool.
        """
        current = (tool_name, tool_input.strip())

        # Strategy 1: Exact Match
        # We count previous occurrences and include the current call when
        # comparing against the threshold so that threshold=2 triggers on the
        # second identical invocation.
        exact_count = sum(
            1 for past_tool, past_input in self.tool_history
            if (past_tool, past_input.strip()) == current
        )

        if exact_count + 1 >= self.exact_threshold:
            self.tool_history.append(current)
            return LoopDetectionResult(
                is_looping=True,
                strategy="exact",
                message=(
                    f"Exact loop detected: '{tool_name}' called {exact_count + 1} "
                    f"times with identical arguments. Change your approach."
                ),
                confidence=1.0,
            )

        # Strategy 2: Fuzzy Match
        # Check against recent history for similar (but not identical) calls
        recent_history = self.tool_history[-5:]  # Last 5 calls
        fuzzy_matches = 0
        for past_tool, past_input in recent_history:
            if past_tool == tool_name:
                similarity = self._jaccard_similarity(tool_input, past_input)
                if similarity >= self.fuzzy_threshold:
                    fuzzy_matches += 1

        if fuzzy_matches >= self.exact_threshold:
            self.tool_history.append(current)
            return LoopDetectionResult(
                is_looping=True,
                strategy="fuzzy",
                message=(
                    f"Fuzzy loop detected: '{tool_name}' called with very similar "
                    f"arguments {fuzzy_matches + 1} times. The rephrasing isn't "
                    f"helping — try a completely different tool or approach."
                ),
                confidence=0.85,
            )

        self.tool_history.append(current)
        return LoopDetectionResult(
            is_looping=False,
            strategy="none",
            message="",
            confidence=0.0,
        )

    def check_output_stagnation(self, output: str) -> LoopDetectionResult:
        """
        Check if the agent's outputs are stagnating
        (producing very similar responses repeatedly).
        """
        self.output_history.append(output)

        if len(self.output_history) < self.stagnation_window:
            return LoopDetectionResult(
                is_looping=False, strategy="none",
                message="", confidence=0.0,
            )

        # Check similarity among the last N outputs
        recent = self.output_history[-self.stagnation_window:]
        similarities = []
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                sim = self._jaccard_similarity(recent[i], recent[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        if avg_similarity >= self.fuzzy_threshold:
            return LoopDetectionResult(
                is_looping=True,
                strategy="stagnation",
                message=(
                    f"Output stagnation detected: last {self.stagnation_window} "
                    f"outputs are {avg_similarity:.0%} similar. The agent is "
                    f"not making progress. Try a different approach entirely."
                ),
                confidence=avg_similarity,
            )

        return LoopDetectionResult(
            is_looping=False, strategy="none",
            message="", confidence=0.0,
        )

    def reset(self):
        self.tool_history.clear()
        self.output_history.clear()
