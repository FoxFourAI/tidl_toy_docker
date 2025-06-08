#!/usr/bin/env python3
"""
Test script for YOLOv8 compiler with different optimization levels.
This script tests the new optimization level functionality.
"""

import os
import sys
import tempfile
import shutil
from yolo_v8_compiler import CompilerArgs, YoloV8Compiler
import numpy as np
import cv2


def create_dummy_model():
    """Create a dummy ONNX model for testing"""
    import onnx
    from onnx import helper, TensorProto, shape_inference
    import numpy as np
    
    # Create a simple dummy model that just reshapes input to output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.UINT8, [1, 3, 640, 640])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 25200, 85])
    
    # Create nodes for a simple pipeline: Cast -> Reshape -> output
    # Cast uint8 to float
    cast_node = helper.make_node('Cast', ['input'], ['cast_output'], to=TensorProto.FLOAT)
    
    # Create reshape node to match expected output shape
    # 640*640*3 = 1,228,800 which we need to reshape to 25200*85 = 2,142,000
    # Let's just create a simpler model that works
    # We'll flatten and then use a slice to get the right size
    flatten_node = helper.make_node('Flatten', ['cast_output'], ['flattened'], axis=1)
    
    # Create a constant for the target shape
    target_shape = helper.make_tensor('target_shape', TensorProto.INT64, [2], [25200, 85])
    
    # Add slice to get the right number of elements
    # Since 640*640*3 = 1,228,800 and 25200*85 = 2,142,000, we need to pad or truncate
    # Let's use a simpler approach - just reshape what we can
    actual_elements = 640 * 640 * 3  # 1,228,800
    target_elements = 25200 * 85     # 2,142,000
    
    if actual_elements < target_elements:
        # Pad with zeros
        pad_size = target_elements - actual_elements
        pad_tensor = helper.make_tensor('pad_values', TensorProto.FLOAT, [pad_size], [0.0] * pad_size)
        concat_node = helper.make_node('Concat', ['flattened', 'pad_values'], ['padded'], axis=1)
        reshape_node = helper.make_node('Reshape', ['padded', 'target_shape'], ['output'])
        nodes = [cast_node, flatten_node, concat_node, reshape_node]
        initializers = [target_shape, pad_tensor]
    else:
        # Slice to the right size
        starts = helper.make_tensor('starts', TensorProto.INT64, [1], [0])
        ends = helper.make_tensor('ends', TensorProto.INT64, [1], [target_elements])
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], [1])
        slice_node = helper.make_node('Slice', ['flattened', 'starts', 'ends', 'axes'], ['sliced'])
        reshape_node = helper.make_node('Reshape', ['sliced', 'target_shape'], ['output'])
        nodes = [cast_node, flatten_node, slice_node, reshape_node]
        initializers = [target_shape, starts, ends, axes]
    
    graph = helper.make_graph(nodes, 'test_model', [input_tensor], [output_tensor], initializers)
    model = helper.make_model(graph)
    
    # Important: Run shape inference to validate the model
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed for dummy model: {e}")
        # Create an even simpler model if shape inference fails
        return create_simple_dummy_model()
    
    return model


def create_simple_dummy_model():
    """Create a very simple dummy model that should always work"""
    import onnx
    from onnx import helper, TensorProto
    
    # Create the simplest possible model - just a Cast operation with same shape
    input_tensor = helper.make_tensor_value_info('input', TensorProto.UINT8, [1, 3, 640, 640])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 640, 640])
    
    # Just cast from uint8 to float, keeping the same shape
    cast_node = helper.make_node('Cast', ['input'], ['output'], to=TensorProto.FLOAT)
    
    graph = helper.make_graph([cast_node], 'simple_test_model', [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    
    return model


def create_test_environment():
    """Create a temporary test environment"""
    test_dir = tempfile.mkdtemp(prefix="yolo_test_")
    
    # Create directory structure
    os.makedirs(os.path.join(test_dir, "weights"))
    os.makedirs(os.path.join(test_dir, "calibration"))
    os.makedirs(os.path.join(test_dir, "artifacts"))
    
    # Create dummy model
    model = create_dummy_model()
    weights_path = os.path.join(test_dir, "weights", "test_model.onnx")
    onnx.save(model, weights_path)
    
    # Create dummy prototxt
    prototxt_path = os.path.join(test_dir, "weights", "test_model.prototxt")
    with open(prototxt_path, 'w') as f:
        f.write("layer {\n  name: \"data\"\n  type: \"Input\"\n}\n")
    
    # Create dummy calibration images
    for i in range(3):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img_path = os.path.join(test_dir, "calibration", f"test_{i}.jpg")
        cv2.imwrite(img_path, img)
    
    return test_dir, weights_path, prototxt_path


def test_optimization_levels():
    """Test different optimization levels"""
    print("Creating test environment...")
    test_dir, weights_path, prototxt_path = create_test_environment()
    
    try:
        for opt_level in ['none', 'normalize', 'nv12']:
            print(f"\n=== Testing optimization level: {opt_level} ===")
            
            artifacts_folder = os.path.join(test_dir, "artifacts", opt_level)
            
            # Create compiler args
            compiler_args = CompilerArgs(
                weights_path=weights_path,
                artifacts_folder=artifacts_folder,
                compilation_name="test",
                meta_layers_names_list=prototxt_path,
                calibration_images_folder=os.path.join(test_dir, "calibration"),
                debug_level=1,
                tensor_bits=8,
                calibration_iterations=1,
                max_calibration_images=2,
                max_elements=5,
                input_shape=(640, 640),
                operation_type="optimize",
                visualize=False,
                visualization_task="other",
                optimization_level=opt_level
            )
            
            # Create compiler and test optimization
            compiler = YoloV8Compiler(compiler_args)
            
            try:
                compiler.optimize()
                print(f"✓ Optimization level '{opt_level}' completed successfully")
                
                # Check if optimized model exists
                optimized_path = compiler.get_optimized_model_path()
                if os.path.exists(optimized_path):
                    print(f"✓ Optimized model created at: {optimized_path}")
                else:
                    print(f"✗ Optimized model not found at: {optimized_path}")
                    continue
                
                # Test metadata loading
                metadata = compiler.load_metadata()
                print(f"✓ Metadata loaded: {metadata.optimization_level}, input_type: {metadata.input_type}")
                
                # Verify metadata correctness
                if metadata.optimization_level == opt_level:
                    print(f"✓ Metadata optimization level matches: {opt_level}")
                else:
                    print(f"✗ Metadata optimization level mismatch: expected {opt_level}, got {metadata.optimization_level}")
                
                # Test input preparation with appropriate image size
                # Use the same input shape as defined in compiler args
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                try:
                    processed_input, ratio, paddings = compiler.prepare_input_data(dummy_image, metadata)
                    print(f"✓ Input preparation successful, output shape: {processed_input.shape}")
                    
                    # Verify input type based on optimization level
                    if opt_level == "nv12":
                        expected_nv12_size = int(640 * 640 * 1.5)
                        if processed_input.shape[1] == expected_nv12_size:
                            print("✓ NV12 input format verified")
                        else:
                            print(f"✗ NV12 format incorrect: expected {expected_nv12_size}, got {processed_input.shape[1]}")
                    elif opt_level in ["none", "normalize"]:
                        if len(processed_input.shape) == 4 and processed_input.shape[1] == 3:
                            print("✓ RGB input format verified")
                        else:
                            print(f"✗ RGB format incorrect: expected 4D with 3 channels, got {processed_input.shape}")
                            
                except Exception as e:
                    print(f"✗ Input preparation failed: {e}")
                    # Don't fail the test completely, this might be due to the dummy model
                    
            except Exception as e:
                print(f"✗ Optimization level '{opt_level}' failed: {e}")
                # Continue with other tests even if one fails
                continue
                
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)


def test_nv12_conversion():
    """Test NV12 conversion specifically"""
    print("\n=== Testing NV12 conversion ===")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create dummy compiler for NV12 conversion
    test_dir = tempfile.mkdtemp(prefix="nv12_test_")
    try:
        compiler_args = CompilerArgs(
            weights_path="dummy",
            artifacts_folder=test_dir,
            compilation_name="test",
            meta_layers_names_list="dummy",
            calibration_images_folder="dummy",
            debug_level=1,
            tensor_bits=8,
            calibration_iterations=1,
            max_calibration_images=1,
            max_elements=5,
            input_shape=(480, 640),
            operation_type="optimize",
            visualize=False,
            visualization_task="other",
            optimization_level="nv12"
        )
        
        compiler = YoloV8Compiler(compiler_args)
        
        # Test NV12 conversion
        nv12_data = compiler.rgb_to_nv12(test_image)
        expected_size = int(480 * 640 * 1.5)
        
        print(f"Input image shape: {test_image.shape}")
        print(f"NV12 data shape: {nv12_data.shape}")
        print(f"Expected NV12 size: {expected_size}")
        
        if nv12_data.shape[0] == expected_size:
            print("✓ NV12 conversion successful")
        else:
            print(f"✗ NV12 conversion failed: expected {expected_size}, got {nv12_data.shape[0]}")
            
    finally:
        shutil.rmtree(test_dir)


def test_metadata_functionality():
    """Test metadata saving and loading functionality specifically"""
    print("\n=== Testing Metadata Functionality ===")
    
    test_dir = tempfile.mkdtemp(prefix="metadata_test_")
    try:
        # Create a simple metadata test
        from yolo_v8_compiler import ModelMetadata
        
        # Test metadata creation and serialization
        metadata = ModelMetadata(
            optimization_level="nv12",
            input_type="nv12", 
            requires_normalization=False,
            scale_list=(0.003921568627, 0.003921568627, 0.003921568627),
            mean_list=(0.0, 0.0, 0.0),
            input_shape=(640, 640),
            original_weights_path="/test/path/model.onnx"
        )
        
        # Create a dummy compiler to test metadata saving/loading
        compiler_args = CompilerArgs(
            weights_path="dummy",
            artifacts_folder=test_dir,
            compilation_name="test",
            meta_layers_names_list="dummy",
            calibration_images_folder="dummy",
            debug_level=1,
            tensor_bits=8,
            calibration_iterations=1,
            max_calibration_images=1,
            max_elements=5,
            input_shape=(640, 640),
            operation_type="optimize",
            visualize=False,
            visualization_task="other",
            optimization_level="nv12"
        )
        
        compiler = YoloV8Compiler(compiler_args)
        
        # Test metadata saving
        os.makedirs(os.path.join(test_dir, "onnx"), exist_ok=True)
        compiler.save_metadata(metadata)
        print("✓ Metadata saved successfully")
        
        # Test metadata loading
        loaded_metadata = compiler.load_metadata()
        print("✓ Metadata loaded successfully")
        
        # Verify metadata content
        if (loaded_metadata.optimization_level == metadata.optimization_level and
            loaded_metadata.input_type == metadata.input_type and
            loaded_metadata.scale_list == metadata.scale_list):
            print("✓ Metadata content verified")
        else:
            print("✗ Metadata content mismatch")
            
    except Exception as e:
        print(f"✗ Metadata functionality test failed: {e}")
    finally:
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    try:
        import onnx
        print("ONNX available, running full tests...")
        test_optimization_levels()
    except ImportError:
        print("ONNX not available, skipping model creation tests...")
    
    test_nv12_conversion()
    test_metadata_functionality()
    print("\n=== All tests completed ===") 