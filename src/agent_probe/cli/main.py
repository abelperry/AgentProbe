"""AgentProbe CLI entry point."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(name="agentprobe", help="AgentProbe — Agent evaluation framework")


def _setup_logging(level: str) -> None:
    """Configure loguru: remove default sink, add one with the desired level."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG, INFO, WARNING, ERROR)"),
) -> None:
    """Run an evaluation experiment."""
    _setup_logging(log_level)

    if not config.exists():
        logger.error("Config file {} not found", config)
        raise typer.Exit(1)

    # Ensure CWD is on sys.path so that benchmarks.* can be imported.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    asyncio.run(_run(config))


async def _run(config_path: Path) -> None:
    from agent_probe.config import EvalExperimentConfig
    from agent_probe.core.factory import ExperimentFactory

    config = EvalExperimentConfig.from_yaml(config_path)
    executor = ExperimentFactory().create(config)
    await executor.run()

    logger.info("Experiment complete.")


if __name__ == "__main__":
    app()
