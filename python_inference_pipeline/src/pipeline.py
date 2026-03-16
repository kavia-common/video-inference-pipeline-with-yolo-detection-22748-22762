from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import PipelineConfig
from src.s3_utils import download_from_s3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameFailure:
    """Represents a frame that failed requirements (no detections or low confidence)."""

    frame_index: int
    timestamp_ms: int
    reason: str
    max_confidence: float
    saved_frame_path: Optional[str]
    saved_label_path: Optional[str]


@dataclass(frozen=True)
class FrameDetectionRow:
    """Row for per-frame CSV."""

    frame_index: int
    timestamp_ms: int
    num_detections: int
    max_confidence: float
    class_counts_json: str
    failure_reason: str


@dataclass(frozen=True)
class PipelineResult:
    """Result object returned by the pipeline."""

    annotated_video_path: str
    per_frame_csv_path: str
    summary_csv_path: str
    total_frames_read: int
    total_frames_processed: int
    total_failures: int
    started_at: str
    finished_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(cfg: PipelineConfig) -> None:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.failed_frames_dir.mkdir(parents=True, exist_ok=True)
    cfg.failed_labels_dir.mkdir(parents=True, exist_ok=True)
    cfg.extracted_frames_dir.mkdir(parents=True, exist_ok=True)


def _resolve_inputs(cfg: PipelineConfig) -> Path:
    """
    Resolve the input video location.

    Priority:
    1) S3_VIDEO_URI (download into WORK_DIR)
    2) INPUT_VIDEO (local file path)
    """
    if cfg.s3_video_uri:
        local_video_path = cfg.work_dir / "input_video"
        # try to keep extension if present
        suffix = Path(cfg.s3_video_uri).suffix
        if suffix:
            local_video_path = local_video_path.with_suffix(suffix)
        logger.info("Downloading video from S3: %s -> %s", cfg.s3_video_uri, local_video_path)
        return download_from_s3(cfg.s3_video_uri, local_video_path)

    if cfg.input_video:
        if not cfg.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {cfg.input_video}")
        return cfg.input_video

    raise ValueError("Either INPUT_VIDEO or S3_VIDEO_URI must be provided.")


def _resolve_model(cfg: PipelineConfig) -> Path:
    """
    Resolve the model weights file.

    - If MODEL_SOURCE starts with s3://, downloads to MODEL_LOCAL_PATH.
    - Otherwise treats MODEL_SOURCE as a local filesystem path and (if different)
      copies it to MODEL_LOCAL_PATH for consistent downstream usage.
    """
    if cfg.model_source.startswith("s3://"):
        logger.info("Downloading model from S3: %s -> %s", cfg.model_source, cfg.model_local_path)
        return download_from_s3(cfg.model_source, cfg.model_local_path)

    src = Path(cfg.model_source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"MODEL_SOURCE not found: {src}")

    if src != cfg.model_local_path:
        cfg.model_local_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a streaming copy for large weights; avoids loading entire file in memory.
        shutil.copy2(src, cfg.model_local_path)

    return cfg.model_local_path


def _draw_boxes(
    frame_bgr: np.ndarray, boxes_xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, names: Dict[int, str]
) -> None:
    for (x1, y1, x2, y2), c, cl in zip(boxes_xyxy.astype(int), conf, cls.astype(int)):
        label = f"{names.get(int(cl), str(cl))} {float(c):.2f}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (59, 130, 246), 2)  # blue-ish
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (17, 24, 39),
            2,
            cv2.LINE_AA,
        )


def _write_yolo_label_file(label_path: Path, frame_shape: Tuple[int, int], boxes_xyxy: np.ndarray, cls: np.ndarray) -> None:
    """
    Write YOLO-format labels (cls x_center y_center w h) normalized to [0,1].
    """
    h, w = frame_shape
    lines: List[str] = []
    for (x1, y1, x2, y2), cl in zip(boxes_xyxy, cls):
        x_center = ((x1 + x2) / 2.0) / w
        y_center = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{int(cl)} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _frame_timestamp_ms(cap: cv2.VideoCapture) -> int:
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    try:
        return int(round(float(pos_ms)))
    except Exception:
        return 0


def _maybe_save_extracted_frame(cfg: PipelineConfig, frame_index: int, frame_bgr: np.ndarray) -> Optional[Path]:
    """
    Save a raw extracted frame to cfg.extracted_frames_dir (JPEG) if configured.

    Returns the saved path, or None if saving is disabled.
    """
    if not cfg.save_extracted_frames:
        return None

    out_path = cfg.extracted_frames_dir / f"frame_{frame_index:06d}.jpg"
    # cv2.imwrite returns bool; if it fails we just warn (do not fail the whole pipeline).
    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        logger.warning("Failed to write extracted frame: %s", out_path)
        return None
    return out_path


# PUBLIC_INTERFACE
def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    """
    Run the full pipeline.

    Steps:
    - Create required folders
    - Download inputs (model/video) from S3 when configured
    - Run YOLO per-frame inference (with stride)
    - Write annotated video
    - Save failed frames and failed labels
    - Write per-frame and summary CSV reports

    Returns
    -------
    PipelineResult
        Paths and counters for produced artifacts.
    """
    started = _utc_now_iso()
    _ensure_dirs(cfg)

    video_path = _resolve_inputs(cfg)
    model_path = _resolve_model(cfg)

    logger.info("Input video: %s", video_path)
    logger.info("Model weights: %s", model_path)
    logger.info("Outputs directory: %s", cfg.output_dir)

    logger.info("Loading YOLO model from: %s", model_path)
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        raise RuntimeError("Unable to determine video frame size.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cfg.annotated_video_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {cfg.annotated_video_path}")

    per_frame_rows: List[FrameDetectionRow] = []
    failures: List[FrameFailure] = []

    total_frames_read = 0
    total_frames_processed = 0

    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        total_frames_read += 1

        if cfg.max_frames is not None and total_frames_read > cfg.max_frames:
            break

        # Optionally save extracted frames locally.
        # - If SAVE_ONLY_PROCESSED_FRAMES=true, we only save frames we actually run inference on.
        # - If false, we save every frame read, including skipped ones.
        if cfg.save_extracted_frames and not cfg.save_only_processed_frames:
            _maybe_save_extracted_frame(cfg, frame_index, frame)

        # Always write original frame if skipping, so output video aligns to input
        if frame_index % cfg.frame_stride != 0:
            writer.write(frame)
            continue

        if cfg.save_extracted_frames and cfg.save_only_processed_frames:
            _maybe_save_extracted_frame(cfg, frame_index, frame)

        total_frames_processed += 1
        timestamp_ms = _frame_timestamp_ms(cap)

        # Ultralytics inference: we pass conf/iou to control NMS and filtering.
        results = model.predict(
            source=frame,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            device=cfg.device,
            verbose=False,
        )

        # results is a list; for single image it contains one item.
        r0 = results[0]
        boxes = r0.boxes

        num_det = 0
        max_conf = 0.0
        class_counts: Dict[int, int] = {}
        failure_reason = ""

        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            failure_reason = "no_detections"
            fail_frame_path = cfg.failed_frames_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(fail_frame_path), frame)
            failures.append(
                FrameFailure(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    reason=failure_reason,
                    max_confidence=0.0,
                    saved_frame_path=str(fail_frame_path),
                    saved_label_path=None,
                )
            )
            writer.write(frame)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            num_det = int(xyxy.shape[0])
            max_conf = float(np.max(conf)) if num_det > 0 else 0.0
            for c in cls.astype(int):
                class_counts[int(c)] = class_counts.get(int(c), 0) + 1

            # If all detections are below threshold (should not happen due to conf=threshold),
            # treat as low-confidence failure anyway for robustness.
            if max_conf < cfg.conf_threshold:
                failure_reason = "low_confidence"
                fail_frame_path = cfg.failed_frames_dir / f"frame_{frame_index:06d}.jpg"
                fail_label_path = cfg.failed_labels_dir / f"frame_{frame_index:06d}.txt"
                cv2.imwrite(str(fail_frame_path), frame)
                _write_yolo_label_file(fail_label_path, (height, width), xyxy, cls)
                failures.append(
                    FrameFailure(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        reason=failure_reason,
                        max_confidence=max_conf,
                        saved_frame_path=str(fail_frame_path),
                        saved_label_path=str(fail_label_path),
                    )
                )
                writer.write(frame)
            else:
                # Draw annotations
                _draw_boxes(frame, xyxy, conf, cls, r0.names)
                writer.write(frame)

        per_frame_rows.append(
            FrameDetectionRow(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                num_detections=num_det,
                max_confidence=max_conf,
                class_counts_json=json.dumps(class_counts, sort_keys=True),
                failure_reason=failure_reason,
            )
        )

    cap.release()
    writer.release()

    # Reports
    df = pd.DataFrame([asdict(r) for r in per_frame_rows])
    cfg.per_frame_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.per_frame_csv_path, index=False)

    # Summary report
    summary = {
        "started_at": started,
        "finished_at": _utc_now_iso(),
        "total_frames_read": total_frames_read,
        "total_frames_processed": total_frames_processed,
        "frame_stride": cfg.frame_stride,
        "conf_threshold": cfg.conf_threshold,
        "iou_threshold": cfg.iou_threshold,
        "total_failures": len(failures),
        "failures_no_detections": sum(1 for f in failures if f.reason == "no_detections"),
        "failures_low_confidence": sum(1 for f in failures if f.reason == "low_confidence"),
    }
    pd.DataFrame([summary]).to_csv(cfg.summary_csv_path, index=False)

    logger.info("Pipeline finished. Outputs at: %s", cfg.output_dir)

    return PipelineResult(
        annotated_video_path=str(cfg.annotated_video_path),
        per_frame_csv_path=str(cfg.per_frame_csv_path),
        summary_csv_path=str(cfg.summary_csv_path),
        total_frames_read=total_frames_read,
        total_frames_processed=total_frames_processed,
        total_failures=len(failures),
        started_at=started,
        finished_at=summary["finished_at"],
    )
