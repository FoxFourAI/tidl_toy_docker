## Installation

### Build Container
```bash
make start force-build=true use-gpu=false
```
* `force-build=true` - rebuilds the image
* `use-gpu=false` - disables GPU support

### Start Container
```bash
make exec
```

### Stop Container
```bash
make stop
```

## Detector
### Prepare Detector weights
```bash
wget "https://foxfour-raw.429eeb224b27f22bcd7d100da8db25db.r2.cloudflarestorage.com/archives/detectors.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=7484dfa201fe4f858056bf7fb83bc392/20250503/us-east-1/s3/aws4_request&X-Amz-Date=20250503T223557Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=92838b5a8f6d842cbd95702bbfe03872b55fa8cfeb217f242ec0e18637b56d32" -O assets/detectors.zip
unzip assets/detectors.zip -d assets
rm -rf assets/detectors.zip assets/__MACOSX
```

### Prepare Calibration data
```bash
wget "https://foxfour-raw.429eeb224b27f22bcd7d100da8db25db.r2.cloudflarestorage.com/archives/coco_calibration_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=7484dfa201fe4f858056bf7fb83bc392/20250504/us-east-1/s3/aws4_request&X-Amz-Date=20250504T111005Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=21fdd285b531f2f262c879b4fc759e97caeab5ae6799dabed9ae044195b8f569" -O assets/coco_calibration_data.zip
unzip assets/coco_calibration_data.zip -d assets/coco_calibration_data
rm assets/coco_calibration_data.zip
```

### Compile & Run Detector with COCO weights (For Output Correctness Check)
```bash
bash ./assets/run_detector_with_coco_weights.sh
```

### Compile & Run Detector with random weights (For Fast Check)
```bash
bash ./assets/run_detector_with_random_weights.sh
```
