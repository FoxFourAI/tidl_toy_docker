# Usage

### Build Container
```bash
make start force-build=true use-gpu=false
```
* `force-build=true` - rebuilds the image
* `use-gpu=false` - disables GPU support (GPU not available right now)

### Start Container
```bash
make exec
```

### Stop Container
```bash
make stop
```

## Compile & Run Custom Detector


### Put artifacts in `assets`
* Put weights in `assets/detectors`
* Put meta layers in `assets/detectors`
* Put calibration data in `assets`
Example file structure:
```
assets/
├── detectors/
│   ├── best_coco_bbox_mAP_epoch_120.onnx
│   └── best_coco_bbox_mAP_epoch_120.prototxt
└── av_calibration_dataset/
```

### Edit `assets/compile_detector_with_av_weights.sh`
* Change `WEIGHTS_PATH` to your weights path
* Change `META_LAYERS_LIST` to your meta layers list path
* Change `CALIBRATION_FOLDER` to your calibration folder
* Change `ARTIFACTS_FOLDER` to your artifacts folder

Important parameters:
* `INPUT_SHAPE` - input shape of the model (736,1280 is recommended)
* `CALIBRATION_ITERATIONS` - number of calibration iterations (20 is recommended)
* `MAX_CALIBRATION_IMAGES` - number of calibration images (50 is recommended)
* `RESIZE_TYPE` - image preprocessing method: "resize" (direct) or "letterbox" (aspect-preserving)

Time of compilation:
`CALIBRATION_ITERATIONS` * `MAX_CALIBRATION_IMAGES` - for testing use 2 (iterations) * 3 (images) = 6 total iterations (~10 minutes, 520.4121 seconds on my local runs)

**1 iteration approximately takes 1.5 minutes, based on this you can calculate the time of compilation for your case.**

For real use:
`CALIBRATION_ITERATIONS` * `MAX_CALIBRATION_IMAGES` * `MAX_ELEMENTS` - for testing use 15 (iterations) * 10 (images) = 150 total iterations (few hours)
You are free to change `CALIBRATION_ITERATIONS` and `MAX_CALIBRATION_IMAGES` for quality/speed trade-off.


Parameters, likely you will not need to change them:
* `TENSOR_BITS` - number of bits for the model (8 bit model is necessary for TI hardware)
* `MAX_ELEMENTS` - number of elements in the model (just for debug)
* `DEBUG_LEVEL` - debug level (log level)
* `COMPILATION_NAME` - name of the compilation (default is `default`)
* `SCALE_LIST` - scale list for the model
* `MEAN_LIST` - mean list for the model
* `OPTIMIZATION_LEVEL` - model optimization level (see below)

### Run in container

```bash
bash ./assets/compile_detector_with_av_weights.sh
bash ./assets/run_detector_with_av_weights.sh
```

# Details

## Image Preprocessing Options

The YOLO v8 compiler now supports two image preprocessing modes via the `RESIZE_TYPE` parameter:

### **"resize" Mode (Direct Resizing) - Default**
- **Usage**: `RESIZE_TYPE="resize"`
- **Behavior**: Direct resize to target dimensions without preserving aspect ratio
- **Advantages**: 
  - Consistent with many model training pipelines
  - **Fixed bounding box alignment issues** (eliminates horizontal/vertical offsets)
  - Better coordinate accuracy for models trained on directly resized images
- **Use when**: Model was trained on directly resized images

### **"letterbox" Mode (Aspect-Preserving)**
- **Usage**: `RESIZE_TYPE="letterbox"`
- **Behavior**: Maintains original aspect ratio with padding (traditional YOLO preprocessing)
- **Advantages**: 
  - Preserves object proportions
  - Traditional YOLO approach
- **Use when**: Model was trained with letterboxing or when object proportion preservation is critical

### Example Configuration:
```bash
# For models trained with direct resizing (recommended for better accuracy)
RESIZE_TYPE="resize"

# For traditional YOLO letterboxing
RESIZE_TYPE="letterbox"
```

## Model Optimization Levels

The YOLO v8 compiler supports different optimization levels for various input formats:

### **"none" - Original Model**
- **Input**: Float RGB images [0,1] 
- **Normalization**: Manual (user responsibility)
- **Use case**: When you want full control over preprocessing
- **Compatible with**: Both `resize` and `letterbox` modes

### **"normalize" - Built-in Normalization (Default)**
- **Input**: uint8 RGB images [0,255]
- **Normalization**: Built into model (automatic)
- **Use case**: Standard RGB input with automatic normalization
- **Compatible with**: Both `resize` and `letterbox` modes

### **"nv12" - TI Hardware Optimized**
- **Input**: NV12 format (Y and UV planes)
- **Normalization**: Built into model (automatic)
- **Conversion**: Automatic RGB→NV12 conversion + normalization
- **Use case**: Optimized for TI hardware accelerators
- **Compatible with**: Both `resize` and `letterbox` modes

### Usage Example:
```bash
# Standard configuration with direct resizing and built-in normalization
OPTIMIZATION_LEVEL="normalize"
RESIZE_TYPE="resize"

# TI hardware optimized with letterboxing
OPTIMIZATION_LEVEL="nv12"  
RESIZE_TYPE="letterbox"

# Original model with direct resizing
OPTIMIZATION_LEVEL="none"
RESIZE_TYPE="resize"
```

## Coordinate Transformation Improvements

Recent updates include **significant improvements to bounding box accuracy**:

- **Fixed horizontal/vertical offsets** that caused bounding boxes to appear slightly displaced
- **Improved coordinate transformation** for direct resizing with separate X/Y scaling factors
- **Enhanced postprocessing** that handles both single ratio (letterbox) and dual ratio (direct resize) modes
- **Backward compatibility** maintained for existing models

### NV12 Format Details
NV12 is a YUV format commonly used in hardware accelerators. When using `optimization_level="nv12"`:
- Input images are automatically converted from RGB to NV12 format using BT.601 full-range conversion
- Built-in normalization is applied after conversion
- Optimized for TI hardware processing with proper coordinate handling

Example configuration:
```bash
WEIGHTS_PATH="${BASE_DIR}/assets/detectors/epoch_1.onnx"
META_LAYERS_LIST="${BASE_DIR}/assets/detectors/epoch_1.prototxt"
CALIBRATION_FOLDER="${BASE_DIR}/assets/vis-drone-sample"
ARTIFACTS_FOLDER="${BASE_DIR}/assets/detector_artifacts/test_run_mmyolo"
COMPILATION_NAME="default"
INPUT_SHAPE="736,1280"
CALIBRATION_ITERATIONS=2
MAX_CALIBRATION_IMAGES=3
MAX_ELEMENTS=5
DEBUG_LEVEL=7
TENSOR_BITS=8
OPTIMIZATION_LEVEL="nv12"  # NV12 input format
RESIZE_TYPE="resize"  # Direct resizing for better accuracy
SCALE_LIST="0.003921568627,0.003921568627,0.003921568627"  # 1/255 for each channel
MEAN_LIST="0.0,0.0,0.0"  # No mean subtraction
```

### Copy-paste edited variables from `assets/compile_detector_with_car_quick_weights.sh` to `assets/run_detector_with_car_quick_weights.sh`

### Compile & Run Detector with AV Weights
```bash
bash ./assets/compile_detector_with_car_quick_weights.sh
bash ./assets/run_detector_with_car_quick_weights.sh
```

### Output file structure

```
assets/
├── detector_artifacts/
│   ├── test_run_mmyolo/
│   │   ├── onnx/                      # ONNX model with optimized operations and preprocessing node
│   │   │   ├── model_with_shapes.onnx # Optimized model with preprocessing
│   │   │   └── config.yaml            # Model metadata and configuration
│   │   ├── onnx_tidl/default/         # Compiled model artifacts folder
│   │   └── visualizations/            # Visualizations of the 8/32-bit model
```

## Notes

* One image inference on PC takes ~1 minute
* **Bounding box accuracy significantly improved** with latest coordinate transformation fixes
* Both preprocessing modes (`resize`/`letterbox`) work with all optimization levels (`none`/`normalize`/`nv12`)
* Default configuration uses `RESIZE_TYPE="resize"` for better coordinate accuracy

## Test Repository (Optional)
### Prepare Detector weights
```bash
wget "https://foxfour-raw.429eeb224b27f22bcd7d100da8db25db.r2.cloudflarestorage.com/archives/detectors.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=7484dfa201fe4f858056bf7fb83bc392/20250503/us-east-1/s3/aws4_request&X-Amz-Date=20250503T223557Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=92838b5a8f6d842cbd95702bbfe03872b55fa8cfeb217f242ec0e18637b56d32" -O assets/detectors.zip
unzip assets/detectors.zip -d assets
rm -rf assets/detectors.zip assets/__MACOSX
```
Note: Available up to 2025-05-10. For new link request author.

### Prepare Calibration data
```bash
wget "https://foxfour-raw.429eeb224b27f22bcd7d100da8db25db.r2.cloudflarestorage.com/archives/av_calibration_dataset.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=7484dfa201fe4f858056bf7fb83bc392/20250601/us-east-1/s3/aws4_request&X-Amz-Date=20250601T122249Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=4aa74310e7d70ff9552babea81c912b10c0ea09f195c784a210ee96551fc1f08" -O assets/av_calibration_dataset.zip
unzip assets/av_calibration_dataset.zip -d assets/av_calibration_dataset
rm assets/av_calibration_dataset.zip
```
Note: Available up to 2025-06-01. For new link request author.

### Compile & Run Detector with AV weights (Autonomous Vehicle Detection)
```bash
bash ./assets/compile_detector_with_av_weights.sh
bash ./assets/run_detector_with_av_weights.sh
```

**Note:** The AV weights scripts use optimized configurations with direct resizing for improved accuracy:

**Compilation script** (`compile_detector_with_av_weights.sh`):
- Weights: `assets/detectors/best_coco_bbox_mAP_epoch_120.onnx`
- Meta layers: `assets/detectors/best_coco_bbox_mAP_epoch_120.prototxt`
- Calibration data: `assets/av_calibration_dataset`
- Artifacts folder: `assets/detector_artifacts/armored-vehicles-detector-nv12-250609-v3`
- Optimization level: `nv12` (TI hardware optimized)
- Resize type: `resize` (direct resizing for better accuracy)
- Calibration iterations: 20
- Max calibration images: 50

**Runtime script** (`run_detector_with_av_weights.sh`):
- Same configuration as compilation script
- Provides both 32-bit and 8-bit model visualizations
- Includes performance measurement capabilities

The scripts now use improved coordinate transformation for more accurate bounding box placement and support both direct resizing and letterboxing modes.
