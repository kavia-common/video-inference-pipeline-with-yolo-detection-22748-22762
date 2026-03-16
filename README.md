# video-inference-pipeline-with-yolo-detection-22748-22762

Python video inference pipeline (YOLO + OpenCV) that:
- Reads a video (local filesystem path or optional S3 download)
- Loads a YOLO weights file (local filesystem path or optional S3 download)
- Produces:
  - `annotated.mp4`
  - `failed_frames/*.jpg`
  - `failed_labels/*.txt` (YOLO label format for low-confidence failures)
  - `per_frame_report.csv`
  - `summary_report.csv`

## Run locally in VS Code (Windows-friendly)

### 1) Create a `.env` file
In `python_inference_pipeline/`, copy `.env.example` to `.env` and set the paths to your local video and model.

Example (Windows paths):
```env
WORK_DIR=./work
OUTPUT_DIR=./outputs
INPUT_VIDEO=C:/data/videos/sample.mp4
MODEL_SOURCE=C:/data/models/best.pt
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
FRAME_STRIDE=1
```

Notes:
- Windows paths like `C:\data\video.mp4` and `C:/data/video.mp4` are supported.
- If your path contains spaces, wrap it in quotes:
  - `INPUT_VIDEO="C:/My Videos/sample.mp4"`

### 2) Install dependencies
From `python_inference_pipeline/`:
```bash
pip install -r requirements.txt
```

### 3) Run the pipeline (local)
From `python_inference_pipeline/`:
```bash
python main.py run
```

Outputs will be written to `OUTPUT_DIR` (default: `WORK_DIR/outputs`).

#### Local extracted frames (optional)
If you want the pipeline to also save extracted frames as JPGs locally, set:
```env
SAVE_EXTRACTED_FRAMES=true
# Save every frame read (including stride-skipped frames) if false:
SAVE_ONLY_PROCESSED_FRAMES=true
```

Frames will be written to:
- `OUTPUT_DIR/extracted_frames/*.jpg`
=======

## Optional: use S3
S3 is optional. If you set these env vars, the pipeline will download inputs from S3:
- `S3_VIDEO_URI=s3://bucket/path/video.mp4`
- `MODEL_SOURCE=s3://bucket/path/model.pt`

To upload outputs, set:
- `S3_OUTPUT_PREFIX=s3://bucket/path/prefix/`

AWS credentials/region must be provided via your normal AWS environment configuration.

## CLI
Entry point:
- `python main.py run --env-file path/to/.env --log-level DEBUG`

## API (optional)
There is a small FastAPI app under `python_inference_pipeline/src/api/` for health/config/run, but the primary usage is the CLI.
