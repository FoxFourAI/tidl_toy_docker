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
ARTIFACTS_FOLDER="${BASE_DIR}/assets/detector_artifacts/yolov8ti-m-736x1280-vehicles-rev-3-250523-nv12"
COMPILATION_NAME="default"
INPUT_SHAPE="736,1280"
CALIBRATION_ITERATIONS=2 # 20
MAX_CALIBRATION_IMAGES=3 # 50
MAX_ELEMENTS=5
DEBUG_LEVEL=7
TENSOR_BITS=8

# NOTE: Those operations would be added to the model by the compiler during optimization phase, input for optimized model: uint8 RGB image
SCALE_LIST="0.003921568627,0.003921568627,0.003921568627"  # 1/255 for each channel
MEAN_LIST="0.0,0.0,0.0"  # No mean subtraction
OPTIMIZATION_LEVEL="nv12"  # "none", "normalize", "nv12"
INFERENCE_IMAGE_INDEX=2  # Index of calibration image to use for visualization (0=first, 1=second, etc.)

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
    --optimization_level "$OPTIMIZATION_LEVEL" \
    --inference_image_index "$INFERENCE_IMAGE_INDEX" \
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
    --optimization_level "$OPTIMIZATION_LEVEL" \
    --inference_image_index "$INFERENCE_IMAGE_INDEX" \
    --operation_type "visualize_32bit" \
    --visualize

# ============================================================
# Compilation Phase
# ============================================================
echo "Starting YOLOv8 compilation with ${INPUT_SHAPE} input shape..."
echo "Using ${MAX_CALIBRATION_IMAGES} calibration images and ${CALIBRATION_ITERATIONS} iterations"

# Remove logs file if it exists
if [ -f "$ARTIFACTS_FOLDER/logs.txt" ]; then
    rm "$ARTIFACTS_FOLDER/logs.txt"
fi

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
    --optimization_level "$OPTIMIZATION_LEVEL" \
    --operation_type "compile" | tee -a "$ARTIFACTS_FOLDER/logs.txt"

echo "Compilation completed successfully!"

# bash ./assets/compile_detector_with_random_weights.sh