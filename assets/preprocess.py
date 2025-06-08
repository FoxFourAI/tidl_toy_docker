from typing import Optional, Tuple, Union
import numpy as np
import cv2


def yolo_resize_with_pad(img, input_size, pad_color: int = 114):
    if len(img.shape) == 3:
        padded_img = np.full(
            (input_size[0], input_size[1], 3), pad_color, dtype=np.uint8
        )
    else:
        padded_img = np.full(input_size, pad_color, dtype=np.uint8)

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    out_w = int(img.shape[1] * r)
    out_h = int(img.shape[0] * r)
    cv2.resize(
        img,
        (out_w, out_h),
        dst=padded_img[:out_h, :out_w],
        interpolation=cv2.INTER_LINEAR,
    )
    return padded_img, r


def get_center_crop(image: np.ndarray, crop_size: int):
    h, w = image.shape[:2]
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)
    return image[top : top + crop_size, left : left + crop_size], (left, top)


def get_center_crop_value(image_shape: Tuple[int, int], center_crop: Optional[Union[float, int]] = None, center_crop_min_size: Optional[int] = None):
    if center_crop is not None:
        if isinstance(center_crop, float):
            center_crop = int(round(max(image_shape) * center_crop))
    if center_crop_min_size is not None:
        center_crop = max(center_crop, center_crop_min_size) if center_crop is not None else center_crop_min_size
    return center_crop


def preprocess(
    img, input_size, swap=(2, 0, 1), normalize: bool = True, color_format="RGB", 
    center_crop: Optional[Union[float, int]] = None, center_crop_min_size: Optional[int] = None,
    mean_list: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale_list: Tuple[float, float, float] = (0.003921568627, 0.003921568627, 0.003921568627)  # 1/255
):
    center_crop = get_center_crop_value(img.shape[:2], center_crop, center_crop_min_size)
    if center_crop is not None:
        img, (x_offset, y_offset) = get_center_crop(img, center_crop)
    if color_format == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    padded_img, r = yolo_resize_with_pad(img, input_size)

    padded_img = padded_img.transpose(swap)
    if normalize:
        # Apply custom normalization: (image + mean) * scale
        padded_img = padded_img.astype(np.float32)
        for c in range(padded_img.shape[0]):
            padded_img[c] = (padded_img[c] + mean_list[c]) * scale_list[c]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    else:
        padded_img = np.ascontiguousarray(padded_img)
    paddings = (x_offset, y_offset) if center_crop is not None else None
    return padded_img, r, paddings
