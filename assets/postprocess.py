import numpy as np


def bboxes_xyxy2xywh(bbox: np.array, inplace: bool = True) -> np.array:
    if not inplace:
        bbox = bbox.copy()
    bbox[:, 2] -= bbox[:, 0]
    bbox[:, 3] -= bbox[:, 1]
    return bbox


def postprocess(dets, ratio, paddings, score_threshold):
    if dets.ndim == 4:
        dets = dets[0, 0]
    if dets.ndim == 3:
        dets = dets[0]
    valid_indices = dets[:, 4] > score_threshold
    dets = dets[valid_indices]
    dets[:, :4] /= ratio
    bboxes = np.round(dets[:, :4].copy()).astype(int)
    bboxes = bboxes_xyxy2xywh(bboxes)
    if paddings is not None:
        bboxes[:, 0] += paddings[0]
        bboxes[:, 1] += paddings[1]
    scores, classes = dets[:, 4], dets[:, 5].copy().astype(int)
    return bboxes, scores, classes
