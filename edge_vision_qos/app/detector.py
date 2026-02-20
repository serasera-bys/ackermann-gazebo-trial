from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class Detector(Protocol):
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...


class MockDetector:
    """Fallback detector for environments without ONNX dependencies/model."""

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        mean_intensity = float(frame_bgr.mean())
        confidence = min(0.95, max(0.15, mean_intensity / 255.0))
        if confidence < 0.22:
            return []
        box = (
            w * 0.25,
            h * 0.25,
            w * 0.75,
            h * 0.75,
        )
        return [Detection(label="object", confidence=confidence, bbox_xyxy=box)]


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        import onnxruntime as ort

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_name = outputs[0].name

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        resized, scale, pad = letterbox(frame_bgr, self.image_size)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]

        raw = self.session.run([self.output_name], {self.input_name: chw})[0]
        preds = normalize_yolo_output(raw)
        dets = postprocess_yolo(preds, self.conf_threshold, self.iou_threshold)
        if not dets:
            return []

        out: list[Detection] = []
        pad_x, pad_y = pad
        for x1, y1, x2, y2, conf, cls_idx in dets:
            x1 = max(0.0, (x1 - pad_x) / scale)
            y1 = max(0.0, (y1 - pad_y) / scale)
            x2 = max(0.0, (x2 - pad_x) / scale)
            y2 = max(0.0, (y2 - pad_y) / scale)
            out.append(
                Detection(
                    label=f"class_{int(cls_idx)}",
                    confidence=float(conf),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return out


def build_detector(model_path: str | None, conf_threshold: float = 0.25) -> Detector:
    if not model_path:
        LOGGER.warning("No model path configured. Using MockDetector.")
        return MockDetector()

    model = Path(model_path)
    if not model.exists():
        LOGGER.warning("Model file not found (%s). Using MockDetector.", model)
        return MockDetector()

    try:
        detector = YoloOnnxDetector(str(model), conf_threshold=conf_threshold)
        LOGGER.info("Loaded ONNX detector from %s", model)
        return detector
    except Exception as exc:  # pragma: no cover - runtime dependency issue path
        LOGGER.warning("Failed to load ONNX detector (%s). Using MockDetector.", exc)
        return MockDetector()


def letterbox(frame: np.ndarray, new_size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = frame.shape[:2]
    scale = min(new_size / max(h, 1), new_size / max(w, 1))
    nh, nw = int(round(h * scale)), int(round(w * scale))

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for non-mock inference") from exc

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def normalize_yolo_output(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[1] in (84, 85):
        arr = np.transpose(arr, (0, 2, 1))
    if arr.ndim != 3:
        raise ValueError(f"Unexpected output shape: {arr.shape}")
    return arr[0]


def postprocess_yolo(
    preds: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> list[tuple[float, float, float, float, float, int]]:
    if preds.size == 0:
        return []
    if preds.shape[1] < 6:
        return []

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep = confs >= conf_threshold
    boxes_xywh = boxes_xywh[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]
    if len(confs) == 0:
        return []

    boxes = xywh_to_xyxy(boxes_xywh)
    keep_idxs = nms(boxes, confs, iou_threshold)
    return [
        (
            float(boxes[i, 0]),
            float(boxes[i, 1]),
            float(boxes[i, 2]),
            float(boxes[i, 3]),
            float(confs[i]),
            int(class_ids[i]),
        )
        for i in keep_idxs
    ]


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.copy(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    idxs = np.argsort(-scores)
    keep: list[int] = []

    while len(idxs) > 0:
        i = int(idxs[0])
        keep.append(i)
        if len(idxs) == 1:
            break

        rest = idxs[1:]
        ious = compute_iou(boxes[i], boxes[rest])
        idxs = rest[ious < iou_threshold]

    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return inter / union
