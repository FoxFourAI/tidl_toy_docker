## Basic Commands

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
* `CALIBRATION_ITERATIONS` - number of calibration iterations
* `MAX_CALIBRATION_IMAGES` - number of calibration images

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

## Model Optimization Levels

The YOLO v8 compiler now supports different optimization levels:

* `"none"` - Keep original model unchanged, requires manual input normalization
* `"normalize"` - Add normalization to ONNX model (default, backward compatible)  
* `"nv12"` - Add NV12 input conversion + normalization for TI hardware

### Usage Example:
```bash
# For standard RGB input with built-in normalization (default)
OPTIMIZATION_LEVEL="normalize"

# For NV12 input format (TI hardware optimized)
OPTIMIZATION_LEVEL="nv12"

# For original model without modifications
OPTIMIZATION_LEVEL="none"
```

### NV12 Format
NV12 is a YUV format commonly used in hardware accelerators. When using `optimization_level="nv12"`:
- Input images are automatically converted from RGB to NV12 format
- Built-in normalization is applied
- Optimized for TI hardware processing

Example:
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
OPTIMIZATION_LEVEL="normalize"  # or "nv12" for NV12 input
SCALE_LIST="0.003921568627,0.003921568627,0.003921568627"  # 1/255 for each channel
MEAN_LIST="0.0,0.0,0.0"  # No mean subtraction
```

### Copy-paste edited variables from `assets/compile_detector_with_av_weights.sh` to `assets/run_detector_with_av_weights.sh`

### Compile & Run Detector with AV Weights
```bash
bash ./assets/compile_detector_with_av_weights.sh
bash ./assets/run_detector_with_av_weights.sh
```

### Output file structure

```
assets/
├── detector_artifacts/
│   ├── test_run_mmyolo/
│   │   ├── onnx/                      # ONNX model with optimized operations and preprocessing node
│   │   ├── onnx_tidl/default/         # Compiled model artifacts folder
│   │   └── visualizations/            # Visualizations of the 8/32-bit model
```

## Notes

* one image inference on PC took like ~1 minute

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

**Note:** The AV weights scripts use different configurations:

**Compilation script** (`compile_detector_with_av_weights.sh`):
- Weights: `assets/detectors/best_coco_bbox_mAP_epoch_120.onnx`
- Meta layers: `assets/detectors/best_coco_bbox_mAP_epoch_120.prototxt`
- Calibration data: `assets/av_calibration_dataset`
- Artifacts folder: `assets/detector_artifacts/yolov8ti-m-736x1280-vehicles-rev-3-250523`
- Calibration iterations: 10 (fast compilation)
- Max calibration images: 25

**Runtime script** (`run_detector_with_av_weights.sh`):
- Weights: `assets/detectors/best_coco_bbox_mAP_epoch_120.onnx`
- Meta layers: `assets/detectors/best_coco_bbox_mAP_epoch_120.prototxt`
- Calibration data: `assets/av_calibration_dataset`
- Artifacts folder: `assets/detector_artifacts/yolov8ti-m-736x1280-vehicles-rev-3-250523`
- Calibration iterations: 10 (higher quality)
- Max calibration images: 25

The compilation script is optimized for faster testing, while the runtime script uses higher quality settings and different model weights for better vehicle detection performance.
