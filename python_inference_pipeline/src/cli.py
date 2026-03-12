from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from src.config import load_config
from src.pipeline import run_pipeline

app = typer.Typer(add_completion=False, help="Video inference pipeline CLI (YOLO + OpenCV).")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


@app.command("run")
def run(
    env_file: Optional[Path] = typer.Option(None, "--env-file", help="Optional path to a .env file."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
) -> None:
    """
    Run the pipeline using configuration from environment variables.

    Outputs are written to OUTPUT_DIR (default: WORK_DIR/outputs).
    """
    _setup_logging(log_level)
    cfg = load_config(env_file=env_file)
    result = run_pipeline(cfg)
    typer.echo(json.dumps(result.__dict__, indent=2))


# PUBLIC_INTERFACE
def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
