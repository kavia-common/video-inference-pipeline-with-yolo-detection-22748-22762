#!/bin/bash
cd /home/kavia/workspace/code-generation/video-inference-pipeline-with-yolo-detection-22748-22762/python_inference_pipeline
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

