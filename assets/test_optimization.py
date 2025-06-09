#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add current directory to path to import yolo_v8_compiler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_v8_compiler import YoloV8Compiler, CompilerArgs, ModelMetadata
from utils import read_image


def parse_detection_output(outputs, confidence_threshold=0.5):
    """Parse detection model outputs and return boxes, scores, classes."""
    detections = []
    
    for output in outputs:
        # Handle different output shapes
        if len(output.shape) == 3 and output.shape[2] >= 6:
            batch_detections = output[0]  # Take first batch
            for detection in batch_detections:
                if len(detection) >= 6:
                    x1, y1, x2, y2, confidence, class_id = detection[:6]
                    if confidence > confidence_threshold:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
    
    return detections


def print_detections(detections, model_name):
    """Print detection results in a readable format."""
    print(f"\n{model_name} Detections:")
    print("-" * 50)
    
    if not detections:
        print("No detections found")
        return
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        print(f"Detection {i+1}:")
        print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Class ID: {det['class_id']}")


def test_optimization_level(temp_dir, original_model_path, prototxt_path, test_image, 
                          optimization_level, input_shape=(736, 1280)):
    """Test a specific optimization level"""
    print(f"\n{'='*60}")
    print(f"Testing optimization level: {optimization_level.upper()}")
    print(f"{'='*60}")
    
    # Create artifacts folder for this optimization level
    artifacts_folder = os.path.join(temp_dir, f"artifacts_{optimization_level}")
    
    # Create compiler args
    compiler_args = CompilerArgs(
        weights_path=original_model_path,
        artifacts_folder=artifacts_folder,
        compilation_name="test",
        meta_layers_names_list=prototxt_path,
        calibration_images_folder=os.path.dirname(test_image),
        debug_level=1,
        tensor_bits=8,
        calibration_iterations=1,
        max_calibration_images=1,
        max_elements=5,
        input_shape=input_shape,
        operation_type="optimize",
        visualize=False,
        visualization_task="other",
        optimization_level=optimization_level
    )
    
    # Create compiler and optimize
    compiler = YoloV8Compiler(compiler_args)
    print(f"Creating {optimization_level} optimized model...")
    compiler.optimize()
    
    # Get optimized model path
    optimized_model_path = compiler.get_optimized_model_path()
    print(f"✓ Optimized model created: {os.path.basename(optimized_model_path)}")
    
    # Load metadata
    metadata = compiler.load_metadata()
    print(f"✓ Metadata loaded: {metadata.optimization_level}, input_type: {metadata.input_type}")
    
    # Test input preparation
    image = read_image(test_image)
    print(f"✓ Test image loaded: {image.shape}")
    
    try:
        processed_input, ratio, paddings = compiler.prepare_input_data(image, metadata)
        print(f"✓ Input preparation successful:")
        if metadata.input_type == "nv12":
            y_data, uv_data = processed_input
            print(f"  - Y input dtype: {y_data.dtype}, shape: {y_data.shape}, range: [{y_data.min():.6f}, {y_data.max():.6f}]")
            print(f"  - UV input dtype: {uv_data.dtype}, shape: {uv_data.shape}, range: [{uv_data.min():.6f}, {uv_data.max():.6f}]")
        else:
            print(f"  - Input dtype: {processed_input.dtype}")
            print(f"  - Input shape: {processed_input.shape}")
            print(f"  - Input range: [{processed_input.min():.6f}, {processed_input.max():.6f}]")
        print(f"  - Ratio: {ratio:.4f}")
        print(f"  - Paddings: {paddings}")
    except Exception as e:
        print(f"✗ Input preparation failed: {e}")
        return None
    
    # Test inference
    try:
        print("Running inference...")
        session = ort.InferenceSession(optimized_model_path)
        input_details = session.get_inputs()
        
        if len(input_details) == 1:
            input_name = input_details[0].name
            input_type = input_details[0].type
            print(f"  - Model input: '{input_name}' ({input_type})")
            outputs = session.run(None, {input_name: processed_input})
        else:
            # Handle two-input format for NV12
            print(f"  - Model inputs: {[(inp.name, inp.type) for inp in input_details]}")
            y_data, uv_data = processed_input
            input_names = [inp.name for inp in input_details]
            outputs = session.run(None, {input_names[0]: y_data, input_names[1]: uv_data})
        
        print(f"  - Output shapes: {[out.shape for out in outputs]}")
        
        # Parse detections
        detections = parse_detection_output(outputs, confidence_threshold=0.3)
        print(f"  - Detections found: {len(detections)}")
        
        # Handle shape and dtype for result tracking
        if metadata.input_type == "nv12":
            y_data, uv_data = processed_input
            result_shape = f"Y:{y_data.shape}, UV:{uv_data.shape}"
            result_dtype = f"Y:{y_data.dtype}, UV:{uv_data.dtype}"
            result_range = (float(min(y_data.min(), uv_data.min())), float(max(y_data.max(), uv_data.max())))
        else:
            result_shape = processed_input.shape
            result_dtype = processed_input.dtype
            result_range = (float(processed_input.min()), float(processed_input.max()))
        
        return {
            'optimization_level': optimization_level,
            'metadata': metadata,
            'input_shape': result_shape,
            'input_dtype': result_dtype,
            'input_range': result_range,
            'detections': detections,
            'model_path': optimized_model_path
        }
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return None


def compare_results(results):
    """Compare results across optimization levels"""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print("\nInput Format Comparison:")
    print("-" * 40)
    for result in results:
        if result:
            dtype_str = str(result['input_dtype']).replace('numpy.', '')
            print(f"{result['optimization_level']:>10}: {dtype_str:<8} {str(result['input_shape']):<20} range: {result['input_range']}")
    
    print("\nModel Expectations:")
    print("-" * 40)
    for result in results:
        if result:
            meta = result['metadata']
            print(f"{meta.optimization_level:>10}: {meta.input_type:<5} input, normalization: {meta.requires_normalization}")
    
    print("\nDetection Results:")
    print("-" * 40)
    for result in results:
        if result:
            detections = result['detections']
            print(f"{result['optimization_level']:>10}: {len(detections)} detections")
            if detections:
                # Show first detection
                for det in detections:
                    print(f"            bbox={det['bbox']}, conf={det['confidence']:.3f}, class={det['class_id']}")
    
    # Check if all models produced similar results
    print("\nValidation:")
    print("-" * 40)
    
    if all(r is not None for r in results):
        print("✓ All optimization levels completed successfully")
        
        # Check input format differences
        dtypes = set(str(r['input_dtype']).replace('numpy.', '') for r in results)
        shapes = set(str(r['input_shape']) for r in results)
        
        print(f"✓ Input data types: {dtypes}")
        print(f"✓ Input shapes: {shapes}")
        
        # Verify expected differences
        none_result = next((r for r in results if r['optimization_level'] == 'none'), None)
        normalize_result = next((r for r in results if r['optimization_level'] == 'normalize'), None)
        nv12_result = next((r for r in results if r['optimization_level'] == 'nv12'), None)
        
        if none_result and none_result['input_dtype'] == np.float32:
            print("✓ NONE uses float32 input (correct)")
        else:
            print("✗ NONE should use float32 input")
            
        if normalize_result and normalize_result['input_dtype'] == np.uint8:
            print("✓ NORMALIZE uses uint8 input (correct)")
        else:
            print("✗ NORMALIZE should use uint8 input")
            
        if nv12_result and "uint8" in str(nv12_result['input_dtype']):
            print("✓ NV12 uses uint8 input (correct)")
        else:
            print("✗ NV12 should use uint8 input")
            
        # Check shape differences
        if nv12_result and "Y:" in str(nv12_result['input_shape']) and "UV:" in str(nv12_result['input_shape']):
            print("✓ NV12 uses separate Y and UV inputs (correct)")
            # Verify channels-last format: Y should be (1, H, W, 1), UV should be (1, H//2, W//2, 2)
            shape_str = str(nv12_result['input_shape'])
            if ", 1)" in shape_str and ", 2)" in shape_str:
                print("✓ NV12 uses channels-last format (correct)")
            else:
                print("✗ NV12 should use channels-last format (batch, height, width, channels)")
        else:
            print("✗ NV12 should use separate Y and UV inputs")
            
        detection_counts = [len(r['detections']) for r in results]
        if len(set(detection_counts)) == 1:
            print("✓ All models found same number of detections")
        else:
            print(f"! Different detection counts: {detection_counts}")
    else:
        print("✗ Some optimization levels failed")


def test_all_optimization_levels():
    """Comprehensive test of all optimization levels using real YOLO model"""
    print("Starting comprehensive optimization level test with REAL YOLO model...")
    
    # Use real model and image paths
    original_model_path = "/home/romanv/tidl_toy_docker/assets/detectors/best_coco_bbox_mAP_epoch_120.onnx"
    prototxt_path = "/home/romanv/tidl_toy_docker/assets/detectors/best_coco_bbox_mAP_epoch_120.prototxt"
    test_image = '/home/romanv/tidl_toy_docker/assets/av_calibration_dataset/armored_vehicles#aboba-2#repeat_0#crop#-2023-10-13-030400_png.rf.b7b5fb5bdf277a1b90bba926abd9de91_00.jpg'
    
    # Verify files exist
    if not os.path.exists(original_model_path):
        print(f"Error: Model file not found: {original_model_path}")
        return
    
    if not os.path.exists(prototxt_path):
        print(f"Error: Prototxt file not found: {prototxt_path}")
        return
        
    if not os.path.exists(test_image):
        print(f"Error: Test image not found: {test_image}")
        return
    
    print(f"Using real YOLO model: {os.path.basename(original_model_path)}")
    print(f"Using prototxt: {os.path.basename(prototxt_path)}")
    print(f"Using test image: {os.path.basename(test_image)}")
    
    # Create temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test each optimization level
        optimization_levels = ["none", "normalize", "nv12"]
        results = []
        
        for opt_level in optimization_levels:
            try:
                result = test_optimization_level(
                    temp_dir, original_model_path, prototxt_path, 
                    test_image, opt_level
                )
                results.append(result)
            except Exception as e:
                print(f"Error testing {opt_level}: {e}")
                import traceback
                traceback.print_exc()
                results.append(None)
        
        # Compare results
        compare_results(results)
        
        print(f"\n{'='*60}")
        print("REAL MODEL TEST COMPLETED")
        print(f"{'='*60}")


if __name__ == "__main__":
    test_all_optimization_levels()
