#!/usr/bin/env python3
"""
Video processing script for YOLOv8 TI model
Processes all MP4 files in clips/ directory and outputs results to clips_out/
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import onnxruntime as ort
import glob

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_v8_compiler import YoloV8Compiler, CompilerArgs
from utils import read_image


class VideoProcessor:
    def __init__(self, model_path, input_size=(736, 1280)):
        """
        Initialize video processor with YOLOv8 TI model
        
        Args:
            model_path (str): Path to the ONNX model
            input_size (tuple): Model input size (height, width)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.class_names = [
            'armored_vehicle', 'artillery', 'tank', 'truck', 'car', 'bus',
            'motorcycle', 'bicycle', 'person', 'building', 'plane', 'helicopter'
        ]
        
        # Initialize the model
        self.session = None
        self._load_model()
        
    def _load_model(self):
        """Load the ONNX model for inference"""
        try:
            print(f"Loading model from: {self.model_path}")
            # Use CPU provider for now, can be changed to TI provider later
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"Model loaded successfully!")
            print(f"Input shape: {self.input_shape}")
            print(f"Output names: {self.output_names}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for model input
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            np.ndarray: Preprocessed frame ready for model input
        """
        # Resize frame to model input size
        resized = cv2.resize(frame, (self.input_size[1], self.input_size[0]))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor
    
    def postprocess_detections(self, outputs, conf_threshold=0.5):
        """
        Postprocess model outputs to get bounding boxes and labels
        
        Args:
            outputs (list): Raw model outputs
            conf_threshold (float): Confidence threshold
            
        Returns:
            list: List of detections [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        detections = []
        
        try:
            # Assuming YOLOv8 output format
            output = outputs[0]  # Take first output
            
            # Handle different output shapes
            if len(output.shape) == 3:
                output = output[0]  # Remove batch dimension
            
            # YOLOv8 format: [batch, 84, 8400] -> transpose to [8400, 84]
            if output.shape[0] < output.shape[1]:
                output = output.T
            
            # Extract box coordinates, confidence, and class probabilities
            boxes = output[:, :4]  # x_center, y_center, width, height
            confidences = output[:, 4:5]  # objectness
            class_probs = output[:, 5:]  # class probabilities
            
            # Get class with highest probability for each detection
            class_ids = np.argmax(class_probs, axis=1)
            class_confidences = np.max(class_probs, axis=1)
            
            # Calculate final confidence (objectness * class_prob)
            final_confidences = (confidences.flatten() * class_confidences)
            
            # Filter by confidence threshold
            valid_detections = final_confidences > conf_threshold
            
            if np.any(valid_detections):
                # Convert center format to corner format
                valid_boxes = boxes[valid_detections]
                valid_confidences = final_confidences[valid_detections]
                valid_class_ids = class_ids[valid_detections]
                
                # Convert from normalized coordinates to pixel coordinates
                for i, (box, conf, class_id) in enumerate(zip(valid_boxes, valid_confidences, valid_class_ids)):
                    x_center, y_center, width, height = box
                    
                    # Convert to pixel coordinates
                    x_center *= self.input_size[1]
                    y_center *= self.input_size[0]
                    width *= self.input_size[1]
                    height *= self.input_size[0]
                    
                    # Convert to corner format
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(x1, self.input_size[1]))
                    y1 = max(0, min(y1, self.input_size[0]))
                    x2 = max(0, min(x2, self.input_size[1]))
                    y2 = max(0, min(y2, self.input_size[0]))
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detections.append([x1, y1, x2, y2, float(conf), int(class_id), class_name])
            
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            print(f"Output shape: {[out.shape for out in outputs]}")
        
        return detections
    
    def draw_detections(self, frame, detections, scale_x=1.0, scale_y=1.0):
        """
        Draw red bounding boxes with white labels on frame
        
        Args:
            frame (np.ndarray): Input frame
            detections (list): List of detections
            scale_x (float): Scale factor for x coordinates
            scale_y (float): Scale factor for y coordinates
            
        Returns:
            np.ndarray: Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = detection
            
            # Scale coordinates back to original frame size
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Draw red bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw white background for text
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_height - baseline),
                (x1 + text_width, y1),
                (255, 255, 255),
                -1
            )
            
            # Draw black text on white background
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return result_frame
    
    def process_video(self, input_path, output_path, json_path):
        """
        Process a single video file
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            json_path (str): Path to JSON results file
        """
        print(f"Processing video: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Calculate scale factors
        scale_x = width / self.input_size[1]
        scale_y = height / self.input_size[0]
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize results dictionary
        results = {}
        
        frame_idx = 0
        progress_bar = tqdm(total=total_frames, desc="Processing frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame for model
                input_tensor = self.preprocess_frame(frame)
                
                # Run inference
                try:
                    outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                    detections = self.postprocess_detections(outputs)
                except Exception as e:
                    print(f"Error during inference on frame {frame_idx}: {e}")
                    detections = []
                
                # Draw detections on frame
                result_frame = self.draw_detections(frame, detections, scale_x, scale_y)
                
                # Write frame to output video
                out.write(result_frame)
                
                # Store detections in results
                frame_detections = []
                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id, class_name = detection
                    # Scale coordinates to original frame size for JSON
                    frame_detections.append({
                        'bbox': [
                            int(x1 * scale_x),
                            int(y1 * scale_y), 
                            int(x2 * scale_x),
                            int(y2 * scale_y)
                        ],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
                
                results[frame_idx] = frame_detections
                
                frame_idx += 1
                progress_bar.update(1)
        
        finally:
            cap.release()
            out.release()
            progress_bar.close()
        
        # Save JSON results
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Video processed successfully!")
        print(f"Output video: {output_path}")
        print(f"JSON results: {json_path}")
        print(f"Total frames processed: {frame_idx}")


def main():
    """Main function to process all videos in clips directory"""
    
    # Model path
    model_path = "detector_artifacts/yolov8ti-m-736x1280-vehicles-rev-3-250523-test-run-5/onnx/best_coco_bbox_mAP_epoch_120_with_shapes.onnx"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Initialize processor
    try:
        processor = VideoProcessor(model_path)
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return
    
    # Find all MP4 files in clips directory
    video_files = glob.glob("clips/*.mp4")
    
    if not video_files:
        print("No MP4 files found in clips/ directory")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    for video_path in video_files:
        try:
            # Generate output paths
            video_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(video_name)[0]
            
            output_video_path = f"clips_out/{name_without_ext}_processed.mp4"
            output_json_path = f"clips_out/{name_without_ext}_detections.json"
            
            # Process video
            processor.process_video(video_path, output_video_path, output_json_path)
            print(f"✅ Completed: {video_name}")
            
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            continue
    
    print("🎉 All videos processed!")


if __name__ == "__main__":
    main() 