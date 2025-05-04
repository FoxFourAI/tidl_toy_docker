#!/bin/bash
# Script to compile YOLOv8 model for TI hardware with COCO weights
# This uses a 736x1280 input resolution

# Set script to exit on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/workdir"
WEIGHTS_PATH="${BASE_DIR}/assets/detectors/best_coco_bbox_mAP_epoch_127.onnx"
META_LAYERS_LIST="${BASE_DIR}/assets/detectors/best_coco_bbox_mAP_epoch_127.prototxt"
CALIBRATION_FOLDER="${BASE_DIR}/assets/coco_calibration_data"
ARTIFACTS_FOLDER="${BASE_DIR}/assets/detector_artifacts/yolov8_736x1280_coco"
COMPILATION_NAME="default"
INPUT_SHAPE="736,1280"
CALIBRATION_ITERATIONS=10
MAX_CALIBRATION_IMAGES=15
MAX_ELEMENTS=5
DEBUG_LEVEL=7
TENSOR_BITS=8

# Check if files and directories exist
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Error: Weights file not found at $WEIGHTS_PATH"
    exit 1
fi

if [ ! -f "$META_LAYERS_LIST" ]; then
    echo "Error: Meta layers list file not found at $META_LAYERS_LIST"
    exit 1
fi

if [ ! -d "$CALIBRATION_FOLDER" ]; then
    echo "Error: Calibration folder not found at $CALIBRATION_FOLDER"
    exit 1
fi

# Create artifacts directory if it doesn't exist
mkdir -p "$ARTIFACTS_FOLDER"

# ============================================================
# Optimization Phase
# ============================================================
echo "Starting YOLOv8 optimization with ${INPUT_SHAPE} input shape..."

# NOTE: This is a PC environment for running the YOLO compiler with installed not custom TIDL onnxruntime, it is for optimization phase
/opt/conda/envs/pc-py310/bin/python3 ${SCRIPT_DIR}/yolo_v8_compiler.py \
    --weights_path "$WEIGHTS_PATH" \
    --artifacts_folder "$ARTIFACTS_FOLDER" \
    --compilation_name "$COMPILATION_NAME" \
    --meta_layers_names_list "$META_LAYERS_LIST" \
    --calibration_images_folder "$CALIBRATION_FOLDER" \
    --input_shape "$INPUT_SHAPE" \
    --calibration_iterations "$CALIBRATION_ITERATIONS" \
    --max_calibration_images "$MAX_CALIBRATION_IMAGES" \
    --max_elements "$MAX_ELEMENTS" \
    --debug_level "$DEBUG_LEVEL" \
    --tensor_bits "$TENSOR_BITS" \
    --operation_type "optimize"

echo "Optimization completed successfully!"

# ============================================================
# Visualization Phase - 32-bit model
# ============================================================
echo "Running visualization on 32-bit model..."
# NOTE: Using PC environment Python for 32-bit model visualization
/opt/conda/envs/pc-py310/bin/python3 ${SCRIPT_DIR}/yolo_v8_compiler.py \
    --weights_path "$WEIGHTS_PATH" \
    --artifacts_folder "$ARTIFACTS_FOLDER" \
    --compilation_name "$COMPILATION_NAME" \
    --meta_layers_names_list "$META_LAYERS_LIST" \
    --calibration_images_folder "$CALIBRATION_FOLDER" \
    --input_shape "$INPUT_SHAPE" \
    --calibration_iterations "$CALIBRATION_ITERATIONS" \
    --max_calibration_images "$MAX_CALIBRATION_IMAGES" \
    --max_elements "$MAX_ELEMENTS" \
    --debug_level "$DEBUG_LEVEL" \
    --tensor_bits "$TENSOR_BITS" \
    --operation_type "visualize_32bit" \
    --visualize

# ============================================================
# Compilation Phase
# ============================================================
echo "Starting YOLOv8 compilation with ${INPUT_SHAPE} input shape..."
echo "Using ${MAX_CALIBRATION_IMAGES} calibration images and ${CALIBRATION_ITERATIONS} iterations"

python3 ${SCRIPT_DIR}/yolo_v8_compiler.py \
    --weights_path "$WEIGHTS_PATH" \
    --artifacts_folder "$ARTIFACTS_FOLDER" \
    --compilation_name "$COMPILATION_NAME" \
    --meta_layers_names_list "$META_LAYERS_LIST" \
    --calibration_images_folder "$CALIBRATION_FOLDER" \
    --input_shape "$INPUT_SHAPE" \
    --calibration_iterations "$CALIBRATION_ITERATIONS" \
    --max_calibration_images "$MAX_CALIBRATION_IMAGES" \
    --max_elements "$MAX_ELEMENTS" \
    --debug_level "$DEBUG_LEVEL" \
    --tensor_bits "$TENSOR_BITS" \
    --operation_type "compile"

echo "Compilation completed successfully!"

# bash ./assets/compile_detector_with_coco_weights.sh