from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration for the video inference pipeline.

    Values are sourced from environment variables (optionally via a .env file).
    """

    # General I/O
    work_dir: Path
    input_video: Optional[Path]
    output_dir: Path

    # Model
    model_source: str  # local path or s3://bucket/key
    model_local_path: Path

    # S3 inputs
    s3_video_uri: Optional[str]
    s3_output_prefix: Optional[str]

    # Inference thresholds
    conf_threshold: float
    iou_threshold: float

    # Output artifacts
    annotated_video_path: Path
    failed_frames_dir: Path
    failed_labels_dir: Path
    per_frame_csv_path: Path
    summary_csv_path: Path

    # Runtime
    device: Optional[str]
    max_frames: Optional[int]
    frame_stride: int


def _as_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def _parse_float(name: str, default: str) -> float:
    val = _get_env(name, default)
    try:
        return float(val) if val is not None else float(default)
    except ValueError as e:
        raise ValueError(f"Invalid float for {name}: {val}") from e


def _parse_int_optional(name: str) -> Optional[int]:
    val = _get_env(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}: {val}") from e


def _parse_int(name: str, default: str) -> int:
    val = _get_env(name, default)
    try:
        return int(val) if val is not None else int(default)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}: {val}") from e


# PUBLIC_INTERFACE
def load_config(env_file: Optional[Path] = None) -> PipelineConfig:
    """
    Load pipeline configuration from environment variables.

    Parameters
    ----------
    env_file:
        Optional path to a .env file. If omitted, python-dotenv will look for a default .env.

    Returns
    -------
    PipelineConfig
        The validated pipeline configuration.

    Environment variables
    ---------------------
    Required/commonly used:
    - WORK_DIR: base working directory
    - OUTPUT_DIR: base output directory
    - MODEL_SOURCE: local path or s3://bucket/key to model weights
    - INPUT_VIDEO: local path to input video (optional if S3_VIDEO_URI provided)
    - S3_VIDEO_URI: s3://bucket/key to input video
    - CONF_THRESHOLD: detection confidence threshold (default 0.25)
    - IOU_THRESHOLD: NMS IoU threshold (default 0.45)

    Optional:
    - DEVICE: e.g. "cpu", "0"
    - MAX_FRAMES: limit frames processed
    - FRAME_STRIDE: process every Nth frame (default 1)
    - S3_OUTPUT_PREFIX: s3://bucket/prefix to upload outputs (not required)
    """
    if env_file is not None:
        load_dotenv(env_file)
    else:
        # Load from default .env if present
        load_dotenv()

    work_dir = _as_path(_get_env("WORK_DIR", "./work") or "./work")
    output_dir = _as_path(_get_env("OUTPUT_DIR", str(work_dir / "outputs")) or str(work_dir / "outputs"))

    input_video = _get_env("INPUT_VIDEO")
    s3_video_uri = _get_env("S3_VIDEO_URI")

    model_source = _get_env("MODEL_SOURCE", "./model.pt") or "./model.pt"
    model_local_path = _as_path(_get_env("MODEL_LOCAL_PATH", str(work_dir / "model.pt")) or str(work_dir / "model.pt"))

    conf = _parse_float("CONF_THRESHOLD", "0.25")
    iou = _parse_float("IOU_THRESHOLD", "0.45")

    device = _get_env("DEVICE")
    max_frames = _parse_int_optional("MAX_FRAMES")
    frame_stride = _parse_int("FRAME_STRIDE", "1")
    if frame_stride <= 0:
        raise ValueError("FRAME_STRIDE must be >= 1")

    annotated_video_path = output_dir / "annotated.mp4"
    failed_frames_dir = output_dir / "failed_frames"
    failed_labels_dir = output_dir / "failed_labels"
    per_frame_csv_path = output_dir / "per_frame_report.csv"
    summary_csv_path = output_dir / "summary_report.csv"

    return PipelineConfig(
        work_dir=work_dir,
        input_video=_as_path(input_video) if input_video else None,
        output_dir=output_dir,
        model_source=model_source,
        model_local_path=model_local_path,
        s3_video_uri=s3_video_uri,
        s3_output_prefix=_get_env("S3_OUTPUT_PREFIX"),
        conf_threshold=conf,
        iou_threshold=iou,
        annotated_video_path=annotated_video_path,
        failed_frames_dir=failed_frames_dir,
        failed_labels_dir=failed_labels_dir,
        per_frame_csv_path=per_frame_csv_path,
        summary_csv_path=summary_csv_path,
        device=device,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
