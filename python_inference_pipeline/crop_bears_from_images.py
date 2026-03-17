#!/usr/bin/env python3
"""
crop_bears_from_images.py

Standalone script to detect bears in images using a trained Ultralytics YOLO model,
crop the detected bear regions with padding, and save the crops to an output folder.

How to use (VS Code friendly):
- Manually edit MODEL_PATH, INPUT_DIR, OUTPUT_DIR below (or use CLI args).
- Run: python crop_bears_from_images.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO


# -----------------------------
# User-editable defaults
# -----------------------------
# NOTE: Per user requirements, these should be easy to manually change in VS Code.
MODEL_PATH = "path/to/your/bear_detector.pt"
INPUT_DIR = "path/to/input_images"
OUTPUT_DIR = "path/to/output_crops"

# Padding ratio around the bounding box. 0.2 = 20% padding on each side.
PADDING_RATIO = 0.18


def _iter_image_files(input_dir: Path) -> Iterable[Path]:
    """Yield image files in input_dir (non-recursive) with common image extensions."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _clamp(val: int, lo: int, hi: int) -> int:
    """Clamp integer val into [lo, hi]."""
    return max(lo, min(hi, val))


def _pad_box_xyxy(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    Expand an (x1,y1,x2,y2) box by padding_ratio and clamp to image bounds.

    Returns integer pixel coordinates (x1,y1,x2,y2) suitable for slicing.
    """
    x1, y1, x2, y2 = xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    px1 = int(round(x1 - pad_x))
    py1 = int(round(y1 - pad_y))
    px2 = int(round(x2 + pad_x))
    py2 = int(round(y2 + pad_y))

    # Clamp: x in [0, w], y in [0, h]
    px1 = _clamp(px1, 0, img_w - 1)
    py1 = _clamp(py1, 0, img_h - 1)
    px2 = _clamp(px2, 0, img_w)
    py2 = _clamp(py2, 0, img_h)

    # Ensure proper ordering and non-empty
    if px2 <= px1:
        px2 = min(img_w, px1 + 1)
    if py2 <= py1:
        py2 = min(img_h, py1 + 1)

    return px1, py1, px2, py2


def _choose_bear_class_ids(model: YOLO, class_name: Optional[str]) -> Optional[Sequence[int]]:
    """
    Map a desired class name to one or more class IDs if present in the model.

    If class_name is None: return None (do not filter by class).
    If class_name provided but not found in model class names: return None (no filtering).
    """
    if not class_name:
        return None

    # Ultralytics YOLO models usually expose model.names as a dict or list-like.
    names = getattr(model, "names", None)
    if names is None:
        return None

    # Normalize to dict[int,str]
    if isinstance(names, dict):
        id_to_name = names
    else:
        # list/tuple -> dict
        id_to_name = {i: n for i, n in enumerate(names)}

    target = class_name.strip().lower()
    matched_ids = [i for i, n in id_to_name.items() if str(n).strip().lower() == target]
    return matched_ids or None


# PUBLIC_INTERFACE
def crop_bears_from_folder(
    model_path: str,
    input_dir: str,
    output_dir: str,
    padding_ratio: float = 0.18,
    confidence: float = 0.25,
    bear_class_name: Optional[str] = "bear",
) -> int:
    """
    Detect bears in all images within input_dir, crop detections (with padding), and save to output_dir.

    Args:
        model_path: Path to a trained Ultralytics YOLO model (e.g., .pt).
        input_dir: Folder containing input images.
        output_dir: Folder to write cropped images into (created if missing).
        padding_ratio: Fractional padding applied to each side of the bounding box (e.g., 0.2 = 20%).
        confidence: Minimum confidence threshold for detections.
        bear_class_name: Class label to filter on (default "bear"). If not found, no class filtering is applied.

    Returns:
        Total number of crops written.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a folder: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path)

    # If the model has a "bear" class, filter to it; otherwise keep all detections.
    class_ids = _choose_bear_class_ids(model, bear_class_name)

    total_crops = 0
    image_files = list(_iter_image_files(in_dir))
    if not image_files:
        print(f"No image files found in {in_dir}")
        return 0

    for img_path in image_files:
        # Read image with OpenCV (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Run YOLO inference
        # - conf: confidence threshold
        # - classes: optional class filtering (list of class IDs)
        results = model.predict(source=img, conf=confidence, classes=class_ids, verbose=False)

        if not results:
            continue

        # Ultralytics returns one result for this image
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue

        xyxy_list = boxes.xyxy.cpu().numpy()  # shape: (N, 4)
        if xyxy_list.size == 0:
            continue

        stem = img_path.stem
        ext = img_path.suffix  # keep original extension

        # If multiple bears are detected, save each crop separately with an index.
        for idx, (x1, y1, x2, y2) in enumerate(xyxy_list.tolist()):
            px1, py1, px2, py2 = _pad_box_xyxy((x1, y1, x2, y2), w, h, padding_ratio)
            crop = img[py1:py2, px1:px2]

            if crop.size == 0:
                continue

            out_name = f"{stem}_{idx}{ext}"
            out_path = out_dir / out_name

            ok = cv2.imwrite(str(out_path), crop)
            if not ok:
                print(f"[WARN] Failed writing crop: {out_path}")
                continue

            total_crops += 1

        print(f"Processed {img_path.name}: saved {len(xyxy_list)} crop(s)")

    print(f"Done. Total crops saved: {total_crops} -> {out_dir}")
    return total_crops


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for convenient overrides without editing constants."""
    p = argparse.ArgumentParser(description="Crop detected bears from images using Ultralytics YOLO + OpenCV.")
    p.add_argument("--model", default=MODEL_PATH, help="Path to trained YOLO model (.pt).")
    p.add_argument("--input", dest="input_dir", default=INPUT_DIR, help="Input folder of images.")
    p.add_argument("--output", dest="output_dir", default=OUTPUT_DIR, help="Output folder for crops (auto-created).")
    p.add_argument(
        "--padding",
        type=float,
        default=PADDING_RATIO,
        help="Padding ratio (e.g. 0.2 adds ~20%% padding around the box).",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    p.add_argument(
        "--class-name",
        default="bear",
        help='Class name to filter on (default: "bear"). If not found in the model, no filtering is applied.',
    )
    return p.parse_args(argv)


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()

    # Expand user home (~) and environment variables for convenience
    model_path = os.path.expandvars(os.path.expanduser(args.model))
    input_dir = os.path.expandvars(os.path.expanduser(args.input_dir))
    output_dir = os.path.expandvars(os.path.expanduser(args.output_dir))

    crop_bears_from_folder(
        model_path=model_path,
        input_dir=input_dir,
        output_dir=output_dir,
        padding_ratio=args.padding,
        confidence=args.conf,
        bear_class_name=args.class_name,
    )


if __name__ == "__main__":
    main()
