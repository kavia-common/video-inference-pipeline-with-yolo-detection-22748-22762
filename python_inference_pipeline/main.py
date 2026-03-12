"""
Container entrypoint for the batch pipeline.

Run:
    python main.py

This delegates to the Typer CLI in src.cli.
"""
from src.cli import main

if __name__ == "__main__":
    main()
