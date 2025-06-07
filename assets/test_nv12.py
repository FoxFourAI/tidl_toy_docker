import cv2
import numpy as np
import onnxruntime as ort
from onnx_model_optimizer import add_nv12_conversion_to_onnx_model


def parse_detection_output(outputs, confidence_threshold=0.5):
    """Parse detection model outputs and return boxes, scores, classes."""
    detections = []
    
    for output in outputs:
        # Assuming output shape is [batch, detections, 6] where 6 = [x1, y1, x2, y2, confidence, class]
        if len(output.shape) == 3 and output.shape[2] == 6:
            batch_detections = output[0]  # Take first batch
            for detection in batch_detections:
                x1, y1, x2, y2, confidence, class_id = detection
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
        print()


def rgb_to_nv12(rgb_image):
    """
    Convert RGB image to NV12 format (simplified).
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
    
    Returns:
        nv12_data: NV12 data as flattened array
    """
    height, width = rgb_image.shape[:2]
    
    # Convert RGB to YUV using OpenCV
    yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    
    # Extract Y channel
    y_plane = yuv[:, :, 0]
    
    # Downsample U and V channels by 2x2
    u_channel = yuv[::2, ::2, 1]  # Subsample by 2
    v_channel = yuv[::2, ::2, 2]  # Subsample by 2
    
    # Interleave U and V to create UV plane for NV12
    uv_height, uv_width = u_channel.shape
    uv_plane = np.empty((uv_height, uv_width, 2), dtype=np.uint8)
    uv_plane[:, :, 0] = u_channel
    uv_plane[:, :, 1] = v_channel
    
    # Flatten and combine Y and UV planes
    y_flat = y_plane.flatten()
    uv_flat = uv_plane.flatten()
    
    # Create NV12 data with expected size
    expected_size = int(height * width * 1.5)
    nv12_data = np.zeros(expected_size, dtype=np.uint8)
    
    # Copy Y plane
    y_size = height * width
    nv12_data[:y_size] = y_flat
    
    # Copy UV plane (pad or truncate if necessary)
    uv_start = y_size
    uv_available = min(len(uv_flat), expected_size - uv_start)
    nv12_data[uv_start:uv_start + uv_available] = uv_flat[:uv_available]
    
    return nv12_data


def test_nv12_conversion(original_model_path, nv12_model_path, test_image_path):
    """
    Test that both original and NV12-modified models run successfully.
    """
    
    # Create NV12 model
    print("Creating NV12 model...")
    add_nv12_conversion_to_onnx_model(original_model_path, nv12_model_path)
    print("✓ NV12 model created")
    
    # Load models
    print("Loading models...")
    original_session = ort.InferenceSession(original_model_path)
    nv12_session = ort.InferenceSession(nv12_model_path)
    print("✓ Models loaded")
    
    # Load test image
    print("Loading test image...")
    image = cv2.imread(test_image_path)
    print(f"✓ Image loaded: {image.shape}")
    
    # Get model shapes
    original_shape = original_session.get_inputs()[0].shape
    nv12_shape = nv12_session.get_inputs()[0].shape
    print(f"Original input: {original_shape}")
    print(f"NV12 input: {nv12_shape}")
    
    # Prepare inputs
    target_height, target_width = original_shape[2], original_shape[3]
    resized_image = cv2.resize(image, (target_width, target_height))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Original model input
    rgb_input = rgb_image.transpose(2, 0, 1)  # HWC to CHW
    rgb_input = np.expand_dims(rgb_input, axis=0).astype(np.uint8)
    
    # NV12 model input
    nv12_data = rgb_to_nv12(rgb_image)
    nv12_input = np.expand_dims(nv12_data, axis=0).astype(np.uint8)
    
    # Run inference
    print("Running inference...")
    
    original_input_name = original_session.get_inputs()[0].name
    original_outputs = original_session.run(None, {original_input_name: rgb_input})
    print(f"✓ Original model: {[out.shape for out in original_outputs]}")
    
    nv12_input_name = nv12_session.get_inputs()[0].name
    nv12_outputs = nv12_session.run(None, {nv12_input_name: nv12_input})
    print(f"✓ NV12 model: {[out.shape for out in nv12_outputs]}")
    
    # Parse and display detection results
    original_detections = parse_detection_output(original_outputs, confidence_threshold=0.3)
    nv12_detections = parse_detection_output(nv12_outputs, confidence_threshold=0.3)
    
    print_detections(original_detections, "ORIGINAL MODEL")
    print_detections(nv12_detections, "NV12 MODEL")
    
    print(f"\n✓ Both models completed successfully!")
    print(f"✓ Original model found {len(original_detections)} detections")
    print(f"✓ NV12 model found {len(nv12_detections)} detections")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test NV12 conversion for ONNX models')
    parser.add_argument('--original-model', type=str, required=True, help='Path to original ONNX model')
    parser.add_argument('--nv12-model', type=str, required=True, help='Path to NV12 ONNX model')
    parser.add_argument('--test-image', type=str, required=True, help='Path to test image')

    args = parser.parse_args()
    test_nv12_conversion(
        original_model_path=args.original_model,
        nv12_model_path=args.nv12_model,
        test_image_path=args.test_image,
    )
