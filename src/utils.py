import json
from typing import Any

def safe_json_loads(json_str: str) -> Any:
    """Safely load JSON string."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
