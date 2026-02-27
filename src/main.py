import asyncio

import structlog
import typer

from src.agent.orchestration import OrchestratorAgent
from src.config import settings
from src.logger import configure_logging

import litellm

# Configure LiteLLM to use OpenRouter (OpenAI-compatible)
if getattr(settings, "openrouter_api_key", ""):
    litellm.api_key = settings.openrouter_api_key
    litellm.api_base = settings.openrouter_base_url

# Configure structured logging once at startup
configure_logging()
logger = structlog.get_logger()

# Typer will treat an app with a single command as a "default command",
# meaning you can invoke it without specifying the command name.  In our
# case we want the user to type `research` explicitly, so we add a dummy
# callback.  Having a callback makes the application behave like a true
# command-group and the subcommand name shows up in the help text.
app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    # global CLI options could go here in the future
):
    """AI Research Agent CLI (multi-agent orchestrator)"""
    # nop; logging and other configuration already happened during import
    pass

@app.command("research")
def research(
    query: str = typer.Argument(..., help="The research query to run."),
    model: str = typer.Option(
        None, "--model", "-m", help="LLM model to use (overrides settings)."
    ),
    max_steps: int = typer.Option(
        settings.max_steps, "--max-steps", help="Max ReAct steps per agent."
    ),
):
    """
    Run the multi-agent research pipeline:
    Planner → Research → Fact-check → (optional retry) → Writer
    """
    resolved_model = model or settings.model_name

    logger.info(
        "starting_research",
        query=query,
        model=resolved_model,
        max_steps=max_steps,
    )

    agent = OrchestratorAgent(model=resolved_model, max_steps=max_steps)
    result = asyncio.run(agent.run(query))

    answer = result.get("answer", "")

    # Remove accidental metadata duplication
    if "Metadata:" in answer:
        answer = answer.split("Metadata:")[0]

    print("\n=== FINAL ANSWER ===")
    print(answer.strip())

    print("\n=== FINAL ANSWER ===")
    print(result.get("answer", ""))
    # Optional: print sources if present
    sources = result.get("sources") or result.get("citations") or []
    if sources:
        print("\n === SOURCES === ")
        for s in sources:
            # Handles either dict or string formats
            if isinstance(s, dict):
                print(f"- {s.get('title','(no title)')}: {s.get('url','')}")
            else:
                print(f"- {s}")
    print("\nMetadata:", result.get("metadata", {}))
    

if __name__ == "__main__":
    app()

