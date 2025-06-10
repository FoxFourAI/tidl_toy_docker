#!/bin/bash
# Script to run YOLOv8 model for TI hardware with car weights
# This uses a 736x1280 input resolution

# Set script to exit on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/workdir"
MODEL_PATH="${BASE_DIR}/assets/detectors/epoch_14.onnx"
PROTOTXT_PATH="${BASE_DIR}/assets/detectors/epoch_14.prototxt"
CALIBRATION_FOLDER="${BASE_DIR}/assets/vis-drone-sample"
ARTIFACTS_FOLDER="${BASE_DIR}/assets/detector_artifacts/civil-vehicles-detector-nv12-250609-v3"
COMPILATION_NAME="default"
INPUT_SHAPE="736,1280"
CALIBRATION_ITERATIONS=30
MAX_CALIBRATION_IMAGES=10
MAX_ELEMENTS=5
DEBUG_LEVEL=7
TENSOR_BITS=8

# NOTE: Those operations would be added to the model by the compiler during optimization phase, input for optimized model: uint8 RGB image
SCALE_LIST="0.003921568627,0.003921568627,0.003921568627"  # 1/255 for each channel
MEAN_LIST="0.0,0.0,0.0"  # No mean subtraction
OPTIMIZATION_LEVEL="nv12"  # Options: "none", "normalize", "nv12"
INFERENCE_IMAGE_INDEX=2    # Index of calibration image to use for visualization

# Check if files and directories exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$PROTOTXT_PATH" ]; then
    echo "Error: Prototxt file not found at $PROTOTXT_PATH"
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
    --weights_path "$MODEL_PATH" \
    --artifacts_folder "$ARTIFACTS_FOLDER" \
    --compilation_name "$COMPILATION_NAME" \
    --meta_layers_names_list "$PROTOTXT_PATH" \
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
# Visualization Phase - 8-bit model
# ============================================================
echo "Running visualization on 8-bit model..."
python3 ${SCRIPT_DIR}/yolo_v8_compiler.py \
    --weights_path "$MODEL_PATH" \
    --artifacts_folder "$ARTIFACTS_FOLDER" \
    --compilation_name "$COMPILATION_NAME" \
    --meta_layers_names_list "$PROTOTXT_PATH" \
    --calibration_images_folder "$CALIBRATION_FOLDER" \
    --input_shape "$INPUT_SHAPE" \
    --calibration_iterations "$CALIBRATION_ITERATIONS" \
    --max_calibration_images "$MAX_CALIBRATION_IMAGES" \
    --max_elements "$MAX_ELEMENTS" \
    --debug_level "$DEBUG_LEVEL" \
    --tensor_bits "$TENSOR_BITS" \
    --optimization_level "$OPTIMIZATION_LEVEL" \
    --inference_image_index "$INFERENCE_IMAGE_INDEX" \
    --operation_type "visualize_8bit" \
    --visualize

# ON DEVICE ONLY
# # ============================================================
# # Measurement Phase
# # ============================================================
# echo "Measuring performance of compiled model..."
# python3 ${SCRIPT_DIR}/yolo_v8_compiler.py \
#     --weights_path "$MODEL_PATH" \
#     --artifacts_folder "$ARTIFACTS_FOLDER" \
#     --compilation_name "$COMPILATION_NAME" \
#     --meta_layers_names_list "$PROTOTXT_PATH" \
#     --calibration_images_folder "$CALIBRATION_FOLDER" \
#     --input_shape "$INPUT_SHAPE" \
#     --calibration_iterations "$CALIBRATION_ITERATIONS" \
#     --max_calibration_images "$MAX_CALIBRATION_IMAGES" \
#     --max_elements "$MAX_ELEMENTS" \
#     --debug_level "$DEBUG_LEVEL" \
#     --tensor_bits "$TENSOR_BITS" \
#     --optimization_level "$OPTIMIZATION_LEVEL" \
#     --inference_image_index "$INFERENCE_IMAGE_INDEX" \
#     --operation_type "measure"

echo "All operations completed successfully!"
echo "Model artifacts are located at: $ARTIFACTS_FOLDER"

# bash ./assets/run_detector_with_car_weights.sh
