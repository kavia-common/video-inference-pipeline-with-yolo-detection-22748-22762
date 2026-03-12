from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

openapi_tags = [
    {"name": "health", "description": "Service health checks."},
    {"name": "pipeline", "description": "Run the video inference pipeline and inspect configuration."},
]

app = FastAPI(
    title="Video Inference Pipeline API",
    description=(
        "Runs a YOLO-based video inference pipeline. "
        "This container is primarily intended to run as a CLI (`python main.py`), "
        "but exposes a minimal HTTP API for orchestration."
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    message: str = Field(..., description="Health status message.")


class ConfigResponse(BaseModel):
    config: Dict[str, Any] = Field(..., description="Effective pipeline configuration (resolved from env).")


class RunRequest(BaseModel):
    env_file: Optional[str] = Field(
        None,
        description="Optional filesystem path to a .env file to load before running the pipeline.",
    )


class RunResponse(BaseModel):
    result: Dict[str, Any] = Field(..., description="Pipeline result summary.")


@app.get(
    "/",
    tags=["health"],
    summary="Health check",
    description="Simple liveness check for the API container.",
    response_model=HealthResponse,
    operation_id="health_check",
)
def health_check() -> HealthResponse:
    """Return basic service liveness."""
    return HealthResponse(message="Healthy")


@app.get(
    "/config",
    tags=["pipeline"],
    summary="Get effective configuration",
    description="Loads and returns the effective pipeline configuration from environment variables.",
    response_model=ConfigResponse,
    operation_id="get_config",
)
def get_config() -> ConfigResponse:
    """Load and return the effective pipeline configuration."""
    try:
        cfg = load_config()
        return ConfigResponse(config=cfg.__dict__)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/run",
    tags=["pipeline"],
    summary="Run pipeline",
    description=(
        "Runs the pipeline synchronously in-process. "
        "For large videos, consider using the CLI instead."
    ),
    response_model=RunResponse,
    operation_id="run_pipeline",
)
def run(req: RunRequest) -> RunResponse:
    """Run the pipeline synchronously and return a result object."""
    try:
        cfg = load_config(env_file=Path(req.env_file) if req.env_file else None)
        result = run_pipeline(cfg)
        return RunResponse(result=result.__dict__)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
