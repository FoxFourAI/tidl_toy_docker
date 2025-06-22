import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import argparse
import shutil
import time
import numpy as np
import onnxruntime
import yaml
import onnx
import onnx.shape_inference

from interface import DetectionResult
from preprocess import preprocess
from postprocess import postprocess
from utils import visualize_detections, read_image, save_image

os.environ["TIDL_RT_PERFSTATS"] = "1"


@dataclass
class CompilerArgs:
    weights_path: str
    artifacts_folder: str
    compilation_name: str
    meta_layers_names_list: str
    calibration_images_folder: str
    debug_level: int
    tensor_bits: int
    calibration_iterations: int
    max_calibration_images: int
    max_elements: int
    input_shape: Tuple[int, int]
    operation_type: str
    visualize: bool
    visualization_task: str
    optimization_level: str = "normalize"  # "none", "normalize", "nv12"
    scale_list: Tuple[float, float, float] = (0.003921568627, 0.003921568627, 0.003921568627)
    mean_list: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    inference_image_index: int = 0  # Index of calibration image to use for visualization

    def get_calibration_images(self) -> List[str]:
        """
        Get a list of calibration images.

        Returns:
            A list of calibration image paths.
        """
        calibration_images = []
        for root, _, files in os.walk(self.calibration_images_folder):
            for file in files:
                if (
                    file.lower().endswith(".jpg")
                    or file.lower().endswith(".png")
                    or file.lower().endswith(".jpeg")
                ):
                    calibration_images.append(os.path.join(root, file))
        return calibration_images[:self.max_calibration_images]

    def get_model_options(self) -> Dict[str, Any]:
        return {
            "tensor_bits": self.tensor_bits,
            "debug_level": self.debug_level,
            "model_type": "OD",
            "object_detection:meta_arch_type": 8,
            "object_detection:meta_layers_names_list": self.meta_layers_names_list,
            "advanced_options:params_16bit_names_list": "", # USUALLY EMPTY
            "advanced_options:output_feature_16bit_names_list": "1,3,175,185,195,180,190,200", # first 2 layers are 16-bit, class layes are 16-bit
            # "advanced_options:output_feature_16bit_names_list": "1,3,172,174,175,182,184,185,192,194,195,180,190,200", # first 2 layers are 16-bit, class layes are 16-bit
            "advanced_options:calibration_iterations": self.calibration_iterations,
            "advanced_options:calibration_frames": len(self.get_calibration_images()),
            # https://github.com/TexasInstruments/edgeai-benchmark/blob/16e57a65e7aa2802a6ac286be297ecc5cad93344/configs/detection.py#L202
            "accuracy_level": 1,
            "advanced_options:add_data_convert_ops": 1,
            "advanced_options:high_resolution_optimization": 1,
            "advanced_options:pre_batchnorm_fold": 1,
            "advanced_options:quantization_scale_type": 4,
            "advanced_options:activation_clipping": 1,
            "advanced_options:weight_clipping": 1,
            "advanced_options:bias_calibration": 1,
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "score_threshold": 0.25,
            "with_p6": False,
            "normalize_inputs": False,
            "input_shape": self.input_shape,
            "swap": (2, 0, 1),
            "log_severity_level": 3,
            "color_format": "RGB",
            "center_crop": None,
            "center_crop_min_size": None,
            "class_id": [0],
            "class_name": ["vehicle"],
            "version": "1.0.1",
        }


@dataclass
class ModelMetadata:
    """Metadata about model optimization"""
    optimization_level: str
    input_type: str  # "rgb", "nv12"
    requires_normalization: bool
    scale_list: Tuple[float, float, float]
    mean_list: Tuple[float, float, float]
    input_shape: Tuple[int, int]
    original_weights_path: str


class YoloV8Compiler:
    def __init__(self, compiler_args: CompilerArgs):
        """
        Initialize YoloV8Compiler.

        Args:
            weights_path: Path to the weights file.
            artifacts_folder: Path to the artifacts folder.
            calibration_images: List of images to calibrate the model on.
            model_options: Dictionary containing model options.
        """
        self.compiler_args = compiler_args
        self.options = {
            "tidl_tools_path": "/home/workdir/tools/AM68A/tidl_tools", # "/home/workdir/tidl_tools",
            "artifacts_folder": os.path.join(
                self.compiler_args.artifacts_folder,
                "onnx_tidl",
                self.compiler_args.compilation_name,
            ),
            **self.compiler_args.get_model_options(),
        }
        self.metadata = None

    def get_optimized_model_path(self):
        out_model_path = os.path.join(
            self.compiler_args.artifacts_folder, "onnx", "model_with_shapes.onnx"
        )
        return out_model_path

    def get_metadata_path(self):
        """Get path for model metadata file"""
        metadata_path = os.path.join(
            self.compiler_args.artifacts_folder, "onnx", "config.yaml"
        )
        return metadata_path

    def save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to YAML file"""
        metadata_path = self.get_metadata_path()
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Get model config for confidence_score
        model_config = self.compiler_args.get_model_config()
        
        metadata_dict = {
            "optimization_level": metadata.optimization_level,
            "input_type": metadata.input_type,
            "requires_normalization": metadata.requires_normalization,
            "scale_list": list(metadata.scale_list),
            "mean_list": list(metadata.mean_list),
            "input_shape": list(metadata.input_shape),
            "original_weights_path": metadata.original_weights_path,
            "class_id": [0],
            "class_name": ["vehicle"],
            "version": "1.0.1",
            "confidence_score": model_config["score_threshold"]
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_dict, f, default_flow_style=False, indent=2)
        print(f"Saved model metadata to: {metadata_path}")

    def load_metadata(self) -> ModelMetadata:
        """Load model metadata from YAML file"""
        if self.metadata is not None:
            return self.metadata
            
        metadata_path = self.get_metadata_path()
        if not os.path.exists(metadata_path):
            # For backward compatibility, assume normalized model
            print(f"No metadata found at {metadata_path}, assuming normalized model")
            self.metadata = ModelMetadata(
                optimization_level="normalize",
                input_type="rgb",
                requires_normalization=False,  # Already normalized in model
                scale_list=self.compiler_args.scale_list,
                mean_list=self.compiler_args.mean_list,
                input_shape=self.compiler_args.input_shape,
                original_weights_path=self.compiler_args.weights_path
            )
            return self.metadata
        
        with open(metadata_path, 'r') as f:
            metadata_dict = yaml.safe_load(f)
        
        self.metadata = ModelMetadata(
            optimization_level=metadata_dict["optimization_level"],
            input_type=metadata_dict["input_type"],
            requires_normalization=metadata_dict["requires_normalization"],
            scale_list=tuple(metadata_dict["scale_list"]),
            mean_list=tuple(metadata_dict["mean_list"]),
            input_shape=tuple(metadata_dict["input_shape"]),
            original_weights_path=metadata_dict["original_weights_path"]
        )
        
        print(f"Loaded model metadata: {self.metadata.optimization_level} optimization, {self.metadata.input_type} input")
        return self.metadata

    def optimize(self):
        """
        Load the model and perform optimization based on optimization_level.
        
        Optimization levels:
        - "none": Copy original model, expects float inputs [0,1], user handles normalization
        - "normalize": Add normalization to model, expects uint8 inputs [0,255], model handles normalization  
        - "nv12": Add NV12→RGB + normalization, expects NV12 uint8 input, model handles conversion & normalization
        """
        if os.path.exists(self.options["artifacts_folder"]):
            raise ValueError(
                "Artifacts folder exists: {}".format(self.options["artifacts_folder"])
            )
        os.makedirs(
            os.path.join(self.compiler_args.artifacts_folder, "onnx"), exist_ok=True
        )
        os.makedirs(self.options["artifacts_folder"])

        out_model_path = self.get_optimized_model_path()
        
        # Ensure the directory for the output model exists
        os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
        
        if os.path.exists(out_model_path):
            print("!!! Optimized model already exists:", out_model_path)
        else:
            print(f"Optimizing model with level: {self.compiler_args.optimization_level}")
            
            if self.compiler_args.optimization_level == "none":
                # Just copy the original model
                shutil.copyfile(self.compiler_args.weights_path, out_model_path)
                print(f"Copied original model to: {out_model_path}")
                
                # Create metadata
                metadata = ModelMetadata(
                    optimization_level="none",
                    input_type="rgb",
                    requires_normalization=True,  # User needs to normalize inputs
                    scale_list=self.compiler_args.scale_list,
                    mean_list=self.compiler_args.mean_list,
                    input_shape=self.compiler_args.input_shape,
                    original_weights_path=self.compiler_args.weights_path
                )
                
            elif self.compiler_args.optimization_level == "normalize":
                from onnx_model_optimizer import add_normalization_to_onnx_model
                print(f"Adding normalization with scale_list={self.compiler_args.scale_list}, mean_list={self.compiler_args.mean_list}")
                add_normalization_to_onnx_model(
                    self.compiler_args.weights_path,
                    out_model_path,
                    scaleList=self.compiler_args.scale_list,
                    meanList=self.compiler_args.mean_list,
                )
                print(f"Created normalized model: {out_model_path}")
                
                # Create metadata
                metadata = ModelMetadata(
                    optimization_level="normalize",
                    input_type="rgb",
                    requires_normalization=False,  # Normalization built into model
                    scale_list=self.compiler_args.scale_list,
                    mean_list=self.compiler_args.mean_list,
                    input_shape=self.compiler_args.input_shape,
                    original_weights_path=self.compiler_args.weights_path
                )
                
            elif self.compiler_args.optimization_level == "nv12":
                from onnx_model_optimizer import add_nv12_normalization
                
                # Single step: NV12 conversion + normalization combined
                print(f"Creating NV12 + normalization model with scale_list={self.compiler_args.scale_list}, mean_list={self.compiler_args.mean_list}")
                add_nv12_normalization(
                    self.compiler_args.weights_path,
                    out_model_path,
                    scaleList=self.compiler_args.scale_list,
                    meanList=self.compiler_args.mean_list,
                )
                
                print(f"Created NV12 + normalized model: {out_model_path}")
                
                # Create metadata
                metadata = ModelMetadata(
                    optimization_level="nv12",
                    input_type="nv12",
                    requires_normalization=False,  # Built into model
                    scale_list=self.compiler_args.scale_list,
                    mean_list=self.compiler_args.mean_list,
                    input_shape=self.compiler_args.input_shape,
                    original_weights_path=self.compiler_args.weights_path
                )
                
            else:
                raise ValueError(f"Unknown optimization_level: {self.compiler_args.optimization_level}")
            
            # Infer shapes in the ONNX model before saving
            print("Inferring shapes in ONNX model...")
            try:
                model = onnx.load(out_model_path)
                model_with_shapes = onnx.shape_inference.infer_shapes(model)
                onnx.save(model_with_shapes, out_model_path)
                print(f"Successfully inferred and saved shapes to: {out_model_path}")
            except Exception as e:
                print(f"Warning: Shape inference failed: {e}")
                print("Continuing with original model...")
            
            # Save metadata
            self.save_metadata(metadata)

        # Handle prototxt file
        proto_model_path = os.path.join(
            self.compiler_args.artifacts_folder, "onnx", "model_with_shapes.prototxt"
        )
        if os.path.exists(proto_model_path):
            print("!!! Prototxt model already exists:", proto_model_path)
        else:
            shutil.copyfile(self.compiler_args.meta_layers_names_list, proto_model_path)
            print("Copied prototxt file to:", proto_model_path)
            
        print("Model path: {}".format(self.compiler_args.weights_path))
        print("Optimized model path: {}".format(out_model_path))
        time.sleep(5)

    def compile(self):
        out_model_path = self.get_optimized_model_path()
        print("Compile optimized model:", out_model_path)

        so = onnxruntime.SessionOptions()
        self.ort_session = onnxruntime.InferenceSession(
            out_model_path,
            providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
            provider_options=[self.options, {}],
            sess_options=so,
        )
        print("Compiled model: {}".format(self.ort_session))

    def calibrate(self):
        """
        Perform inference on calibration images.
        """
        # Load metadata to determine input handling
        metadata = self.load_metadata()
        
        calibration_images = self.compiler_args.get_calibration_images()

        input_details = self.ort_session.get_inputs()
        
        if len(input_details) == 1:
            # Single input model (none/normalize)
            input_detail = input_details[0]
            batch_size, channel_or_size = input_detail.shape[0], input_detail.shape[1]
            print(f"Model input shape: {input_detail.shape}")
            
            assert isinstance(batch_size, str) or batch_size == 1
            
            input_name = input_detail.name
            input_type = input_detail.type
            print(f'Input "{input_name}": {input_type} (metadata: {metadata.input_type})')
        else:
            # Multiple input model (nv12)
            print(f"Model inputs: {[(inp.name, inp.type, inp.shape) for inp in input_details]} (metadata: {metadata.input_type})")

        # Determine expected input type based on metadata
        if metadata.input_type == "nv12":
            # For NV12, check both inputs are uint8
            for inp in input_details:
                assert inp.type == "tensor(uint8)", f"Expected uint8 input, got {inp.type}"
            print("Using NV12 input format (uint8)")
        elif metadata.input_type == "rgb":
            if metadata.requires_normalization:
                print("Using RGB input format with manual normalization")
            else:
                assert input_details[0].type == "tensor(uint8)"
                print("Using RGB input format with built-in normalization")

        for image_path in calibration_images:
            input_data = read_image(image_path)
            
            # Prepare input based on metadata
            processed_input, ratio, paddings = self.prepare_input_data(input_data, metadata)
            
            if metadata.input_type == "nv12":
                # Handle two-input format for NV12
                input_names = [inp.name for inp in self.ort_session.get_inputs()]
                y_data, uv_data = processed_input
                self.ort_session.run(None, {input_names[0]: y_data, input_names[1]: uv_data})
            else:
                input_name = input_details[0].name
                self.ort_session.run(None, {input_name: processed_input})
            
        print("Inference on calibration images complete.")

    def info(self):
        print("Model info:")
        print(self.options)
        time.sleep(10)

    def rgb_to_nv12_planes(self, image: np.ndarray, full_range: bool = True):
        """
        Convert an RGB image (uint8, shape H×W×3) to TI-style NV12 planes.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image of dtype uint8 and shape (H, W, 3).
        full_range : bool, default=True
            • True  → BT.601 full-range (0-255 luma) – what TI TIDL/OpenVX expect.  
            • False → BT.601 *video* (limited range, 16-235 luma).

        Returns
        -------
        y_plane  : np.ndarray  # uint8, shape (1, H,   W,   1) - channels last
        uv_plane : np.ndarray  # uint8, shape (1, H//2, W//2, 2) - channels last (channel 0 = U/Cb, channel 1 = V/Cr)
        """
        # ---------- Sanity checks -------------------------------------------------
        if image.dtype != np.uint8:
            raise ValueError("Input must be uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must have shape (H, W, 3)")
        
        H, W, _ = image.shape
        if (H & 1) or (W & 1):
            raise ValueError("Height and width must be even for 4:2:0 sampling")

        # ---------- Separate channels --------------------------------------------
        R = image[..., 0].astype(np.float32)
        G = image[..., 1].astype(np.float32)
        B = image[..., 2].astype(np.float32)

        # ---------- RGB → YUV -----------------------------------------------------
        if full_range:
            # BT.601 full-range
            Y =  0.29900 * R + 0.58700 * G + 0.11400 * B           # 0-255
            U = -0.168736 * R - 0.331264 * G + 0.500000 * B + 128  # 0-255
            V =  0.500000 * R - 0.418688 * G - 0.081312 * B + 128  # 0-255
        else:
            # BT.601 video-range (16-235 / 16-240)
            Y = ( 0.256788 * R + 0.504129 * G + 0.097906 * B) + 16
            U = (-0.148223 * R - 0.290993 * G + 0.439216 * B) + 128
            V = ( 0.439216 * R - 0.367788 * G - 0.071427 * B) + 128

        Y = np.clip(Y, 0, 255).round().astype(np.uint8)
        U = np.clip(U, 0, 255).round().astype(np.uint8)
        V = np.clip(V, 0, 255).round().astype(np.uint8)

        # ---------- 4:2:0 chroma subsampling (averaging 2×2 blocks) --------------
        U_sub = (
            U[0::2, 0::2].astype(np.uint16) + U[0::2, 1::2].astype(np.uint16) +
            U[1::2, 0::2].astype(np.uint16) + U[1::2, 1::2].astype(np.uint16)
        ) >> 2  # divide by 4
        V_sub = (
            V[0::2, 0::2].astype(np.uint16) + V[0::2, 1::2].astype(np.uint16) +
            V[1::2, 0::2].astype(np.uint16) + V[1::2, 1::2].astype(np.uint16)
        ) >> 2

        # ---------- Pack into requested tensor shapes (channels last format) -----
        y_plane  = Y[None, :, :, None]                              # (1, H,   W,   1)
        uv_plane = np.stack((U_sub.astype(np.uint8),                # (1, H/2, W/2, 2)
                             V_sub.astype(np.uint8)), axis=-1)[None, :, :, :]

        return y_plane, uv_plane

    def rgb_to_nv12(self, rgb_image):
        """
        Convert RGB image to NV12 format with separate Y and UV inputs in channels-last format.
        Uses proper BT.601 full-range conversion for TI hardware compatibility.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
        
        Returns:
            tuple: (y_data, uv_data) where:
                - y_data: Y plane as (1, H, W, 1) uint8, range 0-255
                - uv_data: UV plane as (1, H//2, W//2, 2) uint8, range 0-255
        """
        return self.rgb_to_nv12_planes(rgb_image, full_range=True)

    def prepare_input_data(self, image, metadata: ModelMetadata):
        """
        Prepare input data based on model metadata.
        
        Args:
            image: Input image as numpy array
            metadata: Model metadata
            
        Returns:
            Prepared input data and preprocessing info
        """
        model_config = self.compiler_args.get_model_config()
        
        if metadata.input_type == "nv12":
            # For NV12 models, convert RGB to NV12
            if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
                # First resize to target size, keeping RGB format for NV12 conversion
                import cv2
                
                target_height, target_width = metadata.input_shape
                # Direct resize to target size (no aspect ratio preservation, no padding)
                resized_image = cv2.resize(image, (target_width, target_height))
                
                # For direct resizing, calculate the actual scaling factors
                ratio_x = target_width / image.shape[1]
                ratio_y = target_height / image.shape[0]
                # Return separate ratios for proper coordinate transformation
                ratio = (ratio_x, ratio_y)
                # No padding with direct resizing
                paddings = (0, 0)
                
                # Convert to NV12 (returns tuple of Y and UV data)
                y_data, uv_data = self.rgb_to_nv12(resized_image)
                return (y_data, uv_data), ratio, paddings
            else:
                raise ValueError("NV12 models require RGB input image")
                
        elif metadata.input_type == "rgb":
            # For RGB models - direct resize without padding (consistent with NV12)
            import cv2
            
            target_height, target_width = metadata.input_shape
            
            # Apply center crop if specified
            if model_config["center_crop"] is not None:
                from preprocess import get_center_crop_value, get_center_crop
                center_crop = get_center_crop_value(image.shape[:2], model_config["center_crop"], model_config.get("center_crop_min_size"))
                if center_crop is not None:
                    image, _ = get_center_crop(image, center_crop)
            
            # Convert color format if needed
            if model_config["color_format"] == "BGR":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Direct resize to target size (no aspect ratio preservation, no padding)
            resized_image = cv2.resize(image, (target_width, target_height))
            
            # For direct resizing, calculate the actual scaling factors
            ratio_x = target_width / image.shape[1]
            ratio_y = target_height / image.shape[0]
            # Return separate ratios for proper coordinate transformation
            ratio = (ratio_x, ratio_y)
            # No padding with direct resizing
            paddings = (0, 0)
            
            # Apply channel swapping (typically (2, 0, 1) for RGB -> CHW)
            processed_image = resized_image.transpose(model_config["swap"])
            
            # Apply normalization if required
            if metadata.requires_normalization:  # True for "none", False for "normalize"/"nv12"
                processed_image = processed_image.astype(np.float32)
                for c in range(processed_image.shape[0]):
                    processed_image[c] = (processed_image[c] + metadata.mean_list[c]) * metadata.scale_list[c]
                processed_image = np.ascontiguousarray(processed_image, dtype=np.float32)
            else:
                # For models with built-in normalization, ensure uint8 format
                processed_image = np.ascontiguousarray(processed_image, dtype=np.uint8)
            
            return processed_image[np.newaxis], ratio, paddings
        else:
            raise ValueError(f"Unknown input type: {metadata.input_type}")

    def inference_32_bit(self, image, confidence_threshold=0.2):
        # Load metadata to determine input handling
        metadata = self.load_metadata()
        
        so = onnxruntime.SessionOptions()
        ort_session = onnxruntime.InferenceSession(
            self.get_optimized_model_path(),
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=so,
        )
        input_details = ort_session.get_inputs()
        if len(input_details) == 1:
            input_name = input_details[0].name
            input_type = input_details[0].type
            print(f'Input "{input_name}": {input_type} (metadata: {metadata.input_type})')
        else:
            print(f'Inputs: {[(inp.name, inp.type) for inp in input_details]} (metadata: {metadata.input_type})')

        # Prepare input based on metadata
        processed_input, ratio, paddings = self.prepare_input_data(image, metadata)
        
        if metadata.input_type == "nv12":
            # Handle two-input format for NV12
            input_names = [inp.name for inp in ort_session.get_inputs()]
            y_data, uv_data = processed_input
            outs = ort_session.run(None, {input_names[0]: y_data, input_names[1]: uv_data})
        else:
            input_name = input_details[0].name
            outs = ort_session.run(None, {input_name: processed_input})
        return postprocess(outs[0], ratio, paddings, confidence_threshold)

    def inference_8_bit(self, image, confidence_threshold=0.2):
        # Load metadata to determine input handling
        metadata = self.load_metadata()
        
        options = {
            "tidl_tools_path": "/home/workdir/tools/AM68A/tidl_tools",
            "artifacts_folder": os.path.join(
                self.compiler_args.artifacts_folder,
                "onnx_tidl",
                self.compiler_args.compilation_name,
            ),
        }
        so = onnxruntime.SessionOptions()
        ort_session = onnxruntime.InferenceSession(
            self.get_optimized_model_path(),
            providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[options, {}],
            sess_options=so,
        )
        input_details = ort_session.get_inputs()
        if len(input_details) == 1:
            input_name = input_details[0].name
            input_type = input_details[0].type
            print(f'Input "{input_name}": {input_type} (metadata: {metadata.input_type})')
        else:
            print(f'Inputs: {[(inp.name, inp.type) for inp in input_details]} (metadata: {metadata.input_type})')

        # Prepare input based on metadata
        processed_input, ratio, paddings = self.prepare_input_data(image, metadata)
        
        if metadata.input_type == "nv12":
            # Handle two-input format for NV12
            input_names = [inp.name for inp in ort_session.get_inputs()]
            y_data, uv_data = processed_input
            outs = ort_session.run(None, {input_names[0]: y_data, input_names[1]: uv_data})
        else:
            input_name = input_details[0].name
            outs = ort_session.run(None, {input_name: processed_input})
        return postprocess(outs[0], ratio, paddings, confidence_threshold)


def show_32_bit_model_on_sample(model: YoloV8Compiler, image_path: str) -> None:
    print("Inference 32-bit model on:", image_path)
    image = read_image(image_path)
    bboxes, scores, classes = model.inference_32_bit(image)
    print(bboxes[:model.compiler_args.max_elements])
    print(scores[:model.compiler_args.max_elements])
    print(classes[:model.compiler_args.max_elements])
    result = DetectionResult(bboxes=bboxes, scores=scores, classes=classes)
    
    if model.compiler_args.visualize:
        vis_image = visualize_detections(image, result, task=model.compiler_args.visualization_task)
        vis_image_path = os.path.join(
            model.compiler_args.artifacts_folder,
            "visualizations",
            "32bit_" + os.path.basename(image_path),
        )
        os.makedirs(os.path.dirname(vis_image_path), exist_ok=True)
        save_image(vis_image, vis_image_path)
        print(f"Saved visualization to: {vis_image_path}")
    time.sleep(10)


def show_8_bit_model_on_sample(model: YoloV8Compiler, image_path: str) -> None:
    print("Inference 8-bit model on:", image_path)
    image = read_image(image_path)
    bboxes, scores, classes = model.inference_8_bit(image)
    print(bboxes[:model.compiler_args.max_elements])
    print(scores[:model.compiler_args.max_elements])
    print(classes[:model.compiler_args.max_elements])
    result = DetectionResult(bboxes=bboxes, scores=scores, classes=classes)
    
    if model.compiler_args.visualize:
        vis_image = visualize_detections(image, result, task=model.compiler_args.visualization_task)
        vis_image_path = os.path.join(
            model.compiler_args.artifacts_folder,
            "visualizations",
            "8bit_" + os.path.basename(image_path),
        )
        os.makedirs(os.path.dirname(vis_image_path), exist_ok=True)
        save_image(vis_image, vis_image_path)
        print(f"Saved visualization to: {vis_image_path}")
    time.sleep(10)


def measure_8_bit_model_on_samples(model: YoloV8Compiler, image_paths: List[str]) -> None:
    """
    Measure performance of 8-bit model on multiple images.
    
    Args:
        model: The YoloV8Compiler instance
        image_paths: List of paths to images for measurement
    """
    print(f"Measuring 8-bit model performance on {len(image_paths)} images")
    
    # Load metadata to determine input handling
    metadata = model.load_metadata()
    
    # Load all images into memory
    print("Loading images...")
    images = []
    for image_path in image_paths:
        images.append(read_image(image_path))
    
    # Measure model initialization time
    print("Initializing model...")
    options = {
        "tidl_tools_path": "/home/workdir/tools/AM68A/tidl_tools",
        "artifacts_folder": os.path.join(
            model.compiler_args.artifacts_folder,
            "onnx_tidl",
            model.compiler_args.compilation_name,
        )
    }
    print("Options:")
    print(options)
    start_time = time.time()
    so = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(
        model.get_optimized_model_path(),
        providers=["TIDLExecutionProvider"],
        provider_options=[options],
        sess_options=so,
    )
    init_time = time.time() - start_time
    
    # Get input details
    (input_details,) = ort_session.get_inputs()
    input_name = input_details.name
    
    # Measure inference time for each image
    inference_times = []
    postprocess_times = []
    preprocess_times = []
    for i, image in enumerate(images):
        print(f"Running inference on image {i+1}/{len(images)}...")
        
        # Preprocess image using metadata-aware preprocessing
        start_time = time.time()
        processed_input, ratio, paddings = model.prepare_input_data(image, metadata)
        preprocess_time = time.time() - start_time
        preprocess_times.append(preprocess_time)
        
        # Run inference and measure time
        start_time = time.time()
        outs = ort_session.run(None, {input_name: processed_input})
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Measure postprocessing time
        start_time = time.time()
        bboxes, scores, classes = postprocess(outs[0], ratio, paddings, 0.2)
        postprocess_time = time.time() - start_time
        postprocess_times.append(postprocess_time)
        
        print(f"  Image {i+1}: {len(bboxes)} detections, preprocess: {preprocess_time:.4f}s, inference: {inference_time:.4f}s, postprocess: {postprocess_time:.4f}s")
    
    # Calculate statistics
    mean_infer_time = np.mean(inference_times)
    min_infer_time = np.min(inference_times)
    max_infer_time = np.max(inference_times)
    infer_fps = 1.0 / mean_infer_time
    
    mean_postprocess_time = np.mean(postprocess_times)
    min_postprocess_time = np.min(postprocess_times)
    max_postprocess_time = np.max(postprocess_times)
    
    mean_preprocess_time = np.mean(preprocess_times)
    min_preprocess_time = np.min(preprocess_times)
    max_preprocess_time = np.max(preprocess_times)
    
    mean_total_time = mean_infer_time + mean_postprocess_time + mean_preprocess_time
    total_fps = 1.0 / mean_total_time
    
    # Print results
    print("\nPerformance Results:")
    print(f"Model optimization level: {metadata.optimization_level}")
    print(f"Model input type: {metadata.input_type}")
    print(f"Model initialization time: {init_time:.4f} seconds")
    print(f"Number of images: {len(images)}")
    print("\nInference Time (model execution only):")
    print(f"  Mean: {mean_infer_time:.4f} seconds")
    print(f"  Min: {min_infer_time:.4f} seconds")
    print(f"  Max: {max_infer_time:.4f} seconds")
    print(f"  FPS (inference only): {infer_fps:.2f}")

    print("\nPreprocessing Time:")
    print(f"  Mean: {mean_preprocess_time:.4f} seconds")
    print(f"  Min: {min_preprocess_time:.4f} seconds")
    print(f"  Max: {max_preprocess_time:.4f} seconds")
    
    print("\nPostprocessing Time:")
    print(f"  Mean: {mean_postprocess_time:.4f} seconds")
    print(f"  Min: {min_postprocess_time:.4f} seconds")
    print(f"  Max: {max_postprocess_time:.4f} seconds")
    
    print("\nTotal Time (inference + postprocessing + preprocess):")
    print(f"  Mean: {mean_total_time:.4f} seconds")
    print(f"  FPS (total): {total_fps:.2f}")


def run_operations(model: YoloV8Compiler):
    """
    Run operations based on the specified operation type.
    
    Args:
        model: The YoloV8Compiler instance
    """
    # Get the specified calibration image for testing
    calibration_images = model.compiler_args.get_calibration_images()
    if model.compiler_args.inference_image_index >= len(calibration_images):
        print(f"Warning: inference_image_index {model.compiler_args.inference_image_index} is out of range. Using index 0 instead.")
        image_index = 0
    else:
        image_index = model.compiler_args.inference_image_index
    
    image_path = calibration_images[image_index]
    print(f"Using calibration image {image_index + 1}/{len(calibration_images)}: {os.path.basename(image_path)}")
    
    if model.compiler_args.operation_type == "optimize":
        model.info()
        model.optimize()
    
    elif model.compiler_args.operation_type == "compile":
        start_time = time.time()
        model.compile()
        compile_time = time.time() - start_time
        start_time = time.time()
        model.calibrate()
        calibrate_time = time.time() - start_time
        print(f"Compilation time: {compile_time:.4f} seconds")
        print(f"Calibration time: {calibrate_time:.4f} seconds")


    elif model.compiler_args.operation_type == "visualize_32bit":
        print("\n=== Running inference and visualization on 32-bit model ===")
        show_32_bit_model_on_sample(model, image_path)
    
    elif model.compiler_args.operation_type == "visualize_8bit":
        print("\n=== Running inference and visualization on 8-bit model ===")
        show_8_bit_model_on_sample(model, image_path)
    
    elif model.compiler_args.operation_type == "measure":
        print("\n=== Measuring performance of 8-bit model ===")
        calibration_images = model.compiler_args.get_calibration_images()
        measure_8_bit_model_on_samples(model, calibration_images)


def run_cli(
    weights_path: str,
    artifacts_folder: str,
    compilation_name: str,
    meta_layers_names_list: str,
    calibration_images_folder: str,
    input_shape: Tuple[int, int],
    calibration_iterations: int,
    max_calibration_images: int,
    max_elements: int,
    debug_level: int,
    tensor_bits: int,
    operation_type: str,
    visualize: bool,
    visualization_task: str,
    optimization_level: str = "normalize",
    scale_list: Tuple[float, float, float] = (0.003921568627, 0.003921568627, 0.003921568627),
    mean_list: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    inference_image_index: int = 0
) -> None:
    compiler_args = CompilerArgs(
        weights_path=weights_path,
        artifacts_folder=artifacts_folder,
        compilation_name=compilation_name,
        meta_layers_names_list=meta_layers_names_list,
        calibration_images_folder=calibration_images_folder,
        input_shape=input_shape,
        calibration_iterations=calibration_iterations,
        max_calibration_images=max_calibration_images,
        max_elements=max_elements,
        debug_level=debug_level,
        tensor_bits=tensor_bits,
        operation_type=operation_type,
        visualize=visualize,
        visualization_task=visualization_task,
        optimization_level=optimization_level,
        scale_list=scale_list,
        mean_list=mean_list,
        inference_image_index=inference_image_index
    )
    model = YoloV8Compiler(compiler_args)
    
    # Run operations based on the operation type
    run_operations(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO v8 model compiler and optimizer')
    parser.add_argument('--weights_path', type=str, required=True, 
                        help='Path to the weights file')
    parser.add_argument('--artifacts_folder', type=str, required=True, 
                        help='Path to the artifacts folder')
    parser.add_argument('--compilation_name', type=str, required=True, 
                        help='Name for the compilation')
    parser.add_argument('--meta_layers_names_list', type=str, required=True, 
                        help='Path to meta layers names list file')
    parser.add_argument('--calibration_images_folder', type=str, required=True, 
                        help='Folder containing calibration images')
    parser.add_argument('--input_shape', type=str, required=True,
                        help='Input shape for the model in format "height,width"')
    parser.add_argument('--calibration_iterations', type=int, required=True,
                        help='Number of calibration iterations')
    parser.add_argument('--max_calibration_images', type=int, required=True,
                        help='Maximum number of calibration images to use')
    parser.add_argument('--max_elements', type=int, required=True,
                        help='Maximum number of elements to display')
    parser.add_argument('--debug_level', type=int, required=True,
                        help='Debug level for the compiler')
    parser.add_argument('--tensor_bits', type=int, required=True,
                        help='Number of bits for tensor quantization')
    parser.add_argument('--operation_type', type=str, required=True,
                        choices=['optimize', 'compile', 'visualize_32bit', 'visualize_8bit', 'measure'],
                        help='Operation type to perform')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of detection results')
    parser.add_argument('--visualization_task', type=str, default='other',
                        choices=['other'],
                        help='Visualization task type: airplane, armored_vehicle, or other')
    parser.add_argument('--optimization_level', type=str, default='normalize',
                        choices=['none', 'normalize', 'nv12'],
                        help='Model optimization level: none (original model), normalize (add normalization), nv12 (add NV12 conversion + normalization)')
    parser.add_argument('--scale_list', type=str, default='0.003921568627,0.003921568627,0.003921568627',
                        help='Scale list for model optimization in format "r,g,b"')
    parser.add_argument('--mean_list', type=str, default='0.0,0.0,0.0',
                        help='Mean list for model optimization in format "r,g,b"')
    parser.add_argument('--inference_image_index', type=int, default=0,
                        help='Index of calibration image to use for visualization')
    
    args = parser.parse_args()
    
    # Parse input shape
    try:
        height, width = map(int, args.input_shape.split(','))
        input_shape = (height, width)
    except ValueError:
        print(f"Error: Invalid input_shape format. Expected 'height,width', got '{args.input_shape}'")
        exit(1)
    
    # Parse scale_list and mean_list
    try:
        scale_list = tuple(map(float, args.scale_list.split(',')))
        if len(scale_list) != 3:
            raise ValueError("Scale list must have exactly 3 values")
    except ValueError as e:
        print(f"Error parsing scale_list: {e}")
        exit(1)
        
    try:
        mean_list = tuple(map(float, args.mean_list.split(',')))
        if len(mean_list) != 3:
            raise ValueError("Mean list must have exactly 3 values")
    except ValueError as e:
        print(f"Error parsing mean_list: {e}")
        exit(1)
    
    run_cli(
        weights_path=args.weights_path,
        artifacts_folder=args.artifacts_folder,
        compilation_name=args.compilation_name,
        meta_layers_names_list=args.meta_layers_names_list,
        calibration_images_folder=args.calibration_images_folder,
        input_shape=input_shape,
        calibration_iterations=args.calibration_iterations,
        max_calibration_images=args.max_calibration_images,
        max_elements=args.max_elements,
        debug_level=args.debug_level,
        tensor_bits=args.tensor_bits,
        operation_type=args.operation_type,
        visualize=args.visualize,
        visualization_task=args.visualization_task,
        optimization_level=args.optimization_level,
        scale_list=scale_list,
        mean_list=mean_list,
        inference_image_index=args.inference_image_index
    )
