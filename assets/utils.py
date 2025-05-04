import cv2
import numpy as np
from interface import DetectionResult
import random

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def visualize_detections(image: np.ndarray, detection: DetectionResult, task: str = "other") -> np.ndarray:
    """
    Draw detection bounding boxes on an image.
    
    Args:
        image: Input RGB image
        detection: DetectionResult containing bboxes, scores, and classes
        
    Returns:
        Image with visualized detections
    """
    # Get class names based on task if needed
    if task == "other":
         CLASS_NAMES = None
        
    for i in range(len(detection.bboxes)):
        box = detection.bboxes[i]
        class_id = detection.classes[i]
        score = detection.scores[i]
        
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        
        # Use a color from _COLORS if class_id is in range, otherwise random color
        if 0 <= class_id < len(_COLORS):
            color = (_COLORS[class_id] * 255).astype(np.uint8).tolist()
        else:
            # Generate random color
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        # Format text with class name/id and score
        if CLASS_NAMES is not None and class_id < len(CLASS_NAMES):
            text = f'{CLASS_NAMES[class_id]}:{score:.2f}'
        else:
            text = f'class:{class_id}:{score:.2f}'
        
        # Choose text color based on background color brightness
        txt_color = (0, 0, 0) if sum(color) > 384 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        # Draw text background
        cv2.rectangle(
            image,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            color,
            -1
        )
        cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    
    return image


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from a file path and convert it to RGB.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image in RGB format (not BGR)
    """
    # OpenCV reads images in BGR format
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an RGB image to a file.
    
    Args:
        image: RGB image (not BGR)
        output_path: Path to save the image
    """
    # Convert from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    success = cv2.imwrite(output_path, img_bgr)
    if not success:
        raise ValueError(f"Failed to save image to: {output_path}")
