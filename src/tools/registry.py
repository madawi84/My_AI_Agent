"""
src/tools/registry.py

Tool registry + OpenAI/LiteLLM tool schema generation.

Key fix for your error:
Some providers (often via OpenRouter routes) validate function/tool schemas strictly.
They require:
- "required" MUST exist
- and MUST include EVERY key in "properties"
Even if the Python function parameter has a default (e.g., max_results=5).

So we force:
required = list(properties.keys())
"""

import inspect
from typing import Any, Callable, Dict

from pydantic import BaseModel, create_model


class Tool:
    """A callable tool with an auto-generated Pydantic schema."""

    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
        self.model = self._create_pydantic_model(func)

    def _create_pydantic_model(self, func: Callable) -> type[BaseModel]:
        """
        Create a Pydantic model from a function signature.

        - Parameters without defaults become required: (type, ...)
        - Parameters with defaults become optional: (type, default)
        """
        sig = inspect.signature(func)
        fields: Dict[str, tuple[Any, Any]] = {}

        for param_name, param in sig.parameters.items():
            # Skip instance methods' self
            if param_name == "self":
                continue

            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                annotation = str  # sensible fallback

            default = param.default
            if default == inspect.Parameter.empty:
                # Required field
                fields[param_name] = (annotation, ...)
            else:
                # Optional field with default
                fields[param_name] = (annotation, default)

        return create_model(f"{self.name}Schema", **fields)

    def to_openai_schema(self) -> dict:
        """
        Convert tool to OpenAI / LiteLLM function schema.

        IMPORTANT:
        Some providers require 'required' to include every key in properties.
        This prevents errors like:
        "Invalid schema ... Missing 'max_results'"
        """
        json_schema = self.model.model_json_schema()

        # Pydantic returns:
        # properties = {"query": {...}, "max_results": {...}}
        # required   = ["query"]
        #
        # Some providers demand:
        # required == ["query", "max_results", ...]  (all properties)
        properties = json_schema.get("properties", {}) or {}
        required = list(properties.keys())  # force all keys required

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def execute(self, **kwargs) -> Any:
        """
        Validate args using the Pydantic model, then call the underlying function.
        """
        validated_args = self.model(**kwargs)
        return self.func(**validated_args.model_dump())


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, list[str]] = {}

    def register(self, name: str, description: str, category: str = "general"):
        """
        Decorator to register a function as a tool.

        Usage:
            @registry.register("search_web", "Search the web", category="research")
            def search_web(query: str, max_results: int = 5): ...
        """

        def decorator(func: Callable):
            # 1) Prevent accidental overwrites (common bug during reloads)
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered")

            # 2) Create Tool instance
            tool_obj = Tool(name=name, func=func, description=description)

            # 3) Register in dictionary
            self._tools[name] = tool_obj

            # 4) Add to category map
            if category not in self._categories:
                self._categories[category] = []

            if name not in self._categories[category]:
                self._categories[category].append(name)

            # 5) Return original function (still callable normally)
            return func

        return decorator

    def get_tool(self, name: str) -> Tool | None:
        """Return the Tool by name (or None if missing)."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        """Return all registered Tool objects."""
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> list[Tool]:
        """Return Tool objects belonging to a category."""
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def execute_tool(self, name: str) -> Callable:
        """
        Return the tool's execute method, so an agent can call it.

        Raises KeyError if tool does not exist.
        """
        tool_obj = self.get_tool(name)
        if tool_obj is None:
            raise KeyError(f"Tool '{name}' not found")
        return tool_obj.execute


# Global registry instance
registry = ToolRegistry()