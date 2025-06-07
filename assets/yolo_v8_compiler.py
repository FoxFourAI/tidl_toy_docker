import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import argparse
import shutil
import time
import numpy as np
import onnxruntime

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
    scale_list: Tuple[float, float, float] = (0.003921568627, 0.003921568627, 0.003921568627)
    mean_list: Tuple[float, float, float] = (0.0, 0.0, 0.0)

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
            "advanced_options:add_data_convert_ops": 3,
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
        }


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

    def get_optimized_model_path(self):
        out_model_name = (
            os.path.splitext(os.path.basename(self.compiler_args.weights_path))[0]
            + "_with_shapes.onnx"
        )
        out_model_path = os.path.join(
            self.compiler_args.artifacts_folder, "onnx", out_model_name
        )
        return out_model_path

    def optimize(self):
        """
        Load the model and perform shape inference.
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
        if os.path.exists(out_model_path):
            print("!!! Optimized model already exists:", out_model_path)
        else:
            from onnx_model_optimizer import add_normalization_to_onnx_model, add_nv12_conversion_to_onnx_model
            print(f"Optimizing model with scale_list={self.compiler_args.scale_list}, mean_list={self.compiler_args.mean_list}")
            add_normalization_to_onnx_model(
                self.compiler_args.weights_path,
                out_model_path,
                scaleList=self.compiler_args.scale_list,
                meanList=self.compiler_args.mean_list,
            )
            add_nv12_conversion_to_onnx_model(
                out_model_path,
                out_model_path,
            )
        proto_model_path = out_model_path.rsplit(".", 1)[0] + ".prototxt"
        if os.path.exists(proto_model_path):
            print("!!! Prototxt model already exists:", proto_model_path)
        else:
            shutil.copyfile(self.compiler_args.meta_layers_names_list, proto_model_path)
            print("Copied prototxt file to:", proto_model_path)
        print("Model path: {}".format(self.compiler_args.weights_path))
        print("Inferred model path: {}".format(out_model_path))
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
        calibration_images = self.compiler_args.get_calibration_images()

        (input_details,) = self.ort_session.get_inputs()
        batch_size, channel, height, width = input_details.shape
        print(f"Model input shape: {input_details.shape}")

        assert isinstance(batch_size, str) or batch_size == 1
        assert channel == 3

        input_name = input_details.name
        input_type = input_details.type

        print(f'Input "{input_name}": {input_type}')

        assert input_type == "tensor(uint8)"

        model_config = self.compiler_args.get_model_config()
        for image_path in calibration_images:
            input_data = read_image(image_path)
            image, _, _ = preprocess(
                input_data,
                input_size=model_config["input_shape"],
                swap=model_config["swap"],
                normalize=model_config["normalize_inputs"],
                color_format=model_config["color_format"],
                center_crop=model_config["center_crop"],
            )
            self.ort_session.run(None, {input_name: image[np.newaxis]})
        print("Inference on calibration images complete.")

    def info(self):
        print("Model info:")
        print(self.options)
        time.sleep(10)

    def inference_32_bit(self, image, confidence_threshold=0.2):
        so = onnxruntime.SessionOptions()
        ort_session = onnxruntime.InferenceSession(
            self.get_optimized_model_path(),
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=so,
        )
        (input_details,) = ort_session.get_inputs()
        input_name = input_details.name
        input_type = input_details.type

        print(f'Input "{input_name}": {input_type}')

        model_config = self.compiler_args.get_model_config()
        image, ratio, paddings = preprocess(
                image,
                input_size=model_config["input_shape"],
                swap=model_config["swap"],
                normalize=model_config["normalize_inputs"],
                color_format=model_config["color_format"],
                center_crop=model_config["center_crop"],
            )
        outs = ort_session.run(None, {input_name: image[np.newaxis]})
        return postprocess(outs[0], ratio, paddings, confidence_threshold)

    def inference_8_bit(self, image, confidence_threshold=0.2):
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
        (input_details,) = ort_session.get_inputs()
        input_name = input_details.name
        input_type = input_details.type

        print(f'Input "{input_name}": {input_type}')

        model_config = self.compiler_args.get_model_config()
        image, ratio, paddings = preprocess(
            image,
            input_size=model_config["input_shape"],
            swap=model_config["swap"],
            normalize=model_config["normalize_inputs"],
            color_format=model_config["color_format"],
            center_crop=model_config["center_crop"],
        )
        outs = ort_session.run(None, {input_name: image[np.newaxis]})
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
    model_config = model.compiler_args.get_model_config()
    
    # Measure inference time for each image
    inference_times = []
    postprocess_times = []
    preprocess_times = []
    for i, image in enumerate(images):
        print(f"Running inference on image {i+1}/{len(images)}...")
        # Preprocess image
        start_time = time.time()
        preprocessed_image, ratio, paddings = preprocess(
            image,
            input_size=model_config["input_shape"],
            swap=model_config["swap"],
            normalize=model_config["normalize_inputs"],
            color_format=model_config["color_format"],
            center_crop=model_config["center_crop"],
        )
        preprocess_time = time.time() - start_time
        preprocess_times.append(preprocess_time)
        # Run inference and measure time
        start_time = time.time()
        outs = ort_session.run(None, {input_name: preprocessed_image[np.newaxis]})
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
    # Get the first calibration image for testing
    image_path = model.compiler_args.get_calibration_images()[0]
    
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
    scale_list: Tuple[float, float, float] = (0.003921568627, 0.003921568627, 0.003921568627),
    mean_list: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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
        scale_list=scale_list,
        mean_list=mean_list
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
    parser.add_argument('--scale_list', type=str, default='0.003921568627,0.003921568627,0.003921568627',
                        help='Scale list for model optimization in format "r,g,b"')
    parser.add_argument('--mean_list', type=str, default='0.0,0.0,0.0',
                        help='Mean list for model optimization in format "r,g,b"')
    
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
        scale_list=scale_list,
        mean_list=mean_list
    )
