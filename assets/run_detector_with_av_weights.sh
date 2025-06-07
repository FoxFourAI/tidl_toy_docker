#!/bin/bash
# Script to compile YOLOv8 model for TI hardware with random weights
# This uses a 736x1280 input resolution

# Set script to exit on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/workdir"
WEIGHTS_PATH="${BASE_DIR}/assets/detectors/best_coco_bbox_mAP_epoch_120.onnx"
META_LAYERS_LIST="${BASE_DIR}/assets/detectors/best_coco_bbox_mAP_epoch_120.prototxt"
CALIBRATION_FOLDER="${BASE_DIR}/assets/av_calibration_dataset"
ARTIFACTS_FOLDER="${BASE_DIR}/assets/detector_artifacts/yolov8ti-m-736x1280-vehicles-rev-3-250523-test-run-5"
COMPILATION_NAME="default"
INPUT_SHAPE="736,1280"
CALIBRATION_ITERATIONS=20
MAX_CALIBRATION_IMAGES=50
MAX_ELEMENTS=5
DEBUG_LEVEL=7
TENSOR_BITS=8

# NOTE: Those operations would be added to the model by the compiler during optimization phase, input for optimized model: uint8 RGB image
SCALE_LIST="0.003921568627,0.003921568627,0.003921568627"  # 1/255 for each channel
MEAN_LIST="0.0,0.0,0.0"  # No mean subtraction

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
    --scale_list "$SCALE_LIST" \
    --mean_list "$MEAN_LIST" \
    --visualize

# ============================================================
# Visualization Phase - 8-bit model
# ============================================================
echo "Running visualization on 8-bit model..."
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
    --operation_type "visualize_8bit" \
    --scale_list "$SCALE_LIST" \
    --mean_list "$MEAN_LIST" \
    --visualize

# ON DEVICE ONLY
# # ============================================================
# # Measurement Phase
# # ============================================================
# echo "Measuring performance of compiled model..."
# python3 ${SCRIPT_DIR}/yolo_v8_compiler.py \
#     --weights_path "$WEIGHTS_PATH" \
#     --artifacts_folder "$ARTIFACTS_FOLDER" \
#     --compilation_name "$COMPILATION_NAME" \
#     --meta_layers_names_list "$META_LAYERS_LIST" \
#     --calibration_images_folder "$CALIBRATION_FOLDER" \
#     --input_shape "$INPUT_SHAPE" \
#     --calibration_iterations "$CALIBRATION_ITERATIONS" \
#     --max_calibration_images "$MAX_CALIBRATION_IMAGES" \
#     --max_elements "$MAX_ELEMENTS" \
#     --debug_level "$DEBUG_LEVEL" \
#     --tensor_bits "$TENSOR_BITS" \
#     --operation_type "measure" \
#     --scale_list "$SCALE_LIST" \
#     --mean_list "$MEAN_LIST"

echo "All operations completed successfully!"
echo "Model artifacts are located at: $ARTIFACTS_FOLDER"

# bash ./assets/run_detector_with_random_weights.sh