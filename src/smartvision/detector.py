"""YOLO person and PPE detector.

People come from a pretrained COCO model unless ``model.person_source`` is
``ppe_model``. Helmet, vest, gloves, boots, and goggles come from
``models/best.pt``. The PPE checkpoint's person class is ignored in COCO
mode because its recall is low. Thresholds, image size, and device live in
``config/settings.yaml``. This module accepts a numpy frame and does not
open a camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from smartvision.training import select_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Detection:
    """One person or PPE box.

    Attributes:
        class_name: Canonical label from ``settings.classes``.
        confidence: Detector score in ``[0, 1]``.
        bbox_xyxy: Box as ``(x1, y1, x2, y2)`` in pixels.
    """

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def resolve_device(requested: str) -> str:
    """Resolve a settings device string.

    ``auto`` probes MPS with a real allocation, then CUDA, then CPU. A
    failed MPS probe is skipped so machines that cannot use MPS still run.
    Any other value is returned unchanged.

    Args:
        requested: ``auto``, ``cpu``, ``cuda``, ``mps``, or a device index.

    Returns:
        Device string passed to ultralytics.
    """
    choice = requested.strip().lower()
    if choice != "auto":
        return select_device(requested, mps_available=False, cuda_available=False)
    import torch

    mps_ok = False
    if torch.backends.mps.is_built():
        try:
            torch.zeros(1, device="mps")
            mps_ok = True
        except Exception:
            mps_ok = False
    return select_device(
        "auto",
        mps_available=mps_ok,
        cuda_available=torch.cuda.is_available(),
    )


def class_threshold(
    class_name: str,
    class_confidence: Mapping[str, float],
    default: float,
) -> float:
    """Return the confidence cutoff for one class.

    Args:
        class_name: Canonical label.
        class_confidence: Per-class overrides from settings.
        default: Global confidence used when the class has no override.

    Returns:
        The cutoff in ``[0, 1]``.
    """
    if class_name in class_confidence:
        return float(class_confidence[class_name])
    return float(default)


def detections_from_boxes(
    boxes: Any,
    names: Mapping[int, str],
    *,
    class_confidence: Mapping[str, float],
    default_confidence: float,
    reject_names: set[str],
    allow_ids: set[int] | None = None,
    rename_ids: Mapping[int, str] | None = None,
) -> list[Detection]:
    """Turn a model result's boxes into detections.

    Args:
        boxes: Ultralytics boxes, or ``None`` when the model found nothing.
            Must expose ``xyxy``, ``conf``, and ``cls`` arrays.
        names: Map of class index to model label.
        class_confidence: Per-class cutoffs.
        default_confidence: Cutoff for labels without an override.
        reject_names: Labels dropped after renaming, such as the PPE
            model's person class.
        allow_ids: When set, every other class index is dropped.
        rename_ids: Optional replacement labels for selected class indexes.

    Returns:
        Detections that passed the cutoff. Empty when ``boxes`` is missing.
    """
    if boxes is None:
        return []
    xyxy = _as_array(boxes.xyxy)
    conf = _as_array(boxes.conf)
    cls = _as_array(boxes.cls)
    if xyxy.size == 0:
        return []
    found: list[Detection] = []
    renames = rename_ids or {}
    for row, score, class_id in zip(xyxy, conf, cls):
        index = int(class_id)
        if allow_ids is not None and index not in allow_ids:
            continue
        label = renames.get(index, str(names.get(index, index)))
        if label in reject_names:
            continue
        cutoff = class_threshold(label, class_confidence, default_confidence)
        confidence = float(score)
        if confidence < cutoff:
            continue
        x1, y1, x2, y2 = (float(value) for value in row)
        found.append(Detection(label, confidence, (x1, y1, x2, y2)))
    return found


def _as_array(value: Any) -> np.ndarray:
    """Return a numpy view of a tensor or array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _weights_path(configured: str, root: Path) -> Path:
    """Resolve a weights path against the project root when it is relative."""
    path = Path(configured)
    if not path.is_absolute():
        path = root / path
    return path


class PPEDetector:
    """Run the PPE checkpoint and, in COCO mode, the person checkpoint."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        ppe_model: Any | None = None,
        person_model: Any | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Load both models and warm them up.

        Args:
            settings: Parsed ``config/settings.yaml``.
            ppe_model: Prebuilt PPE model. Production leaves this empty so
                ``models/best.pt`` is loaded.
            person_model: Prebuilt COCO model. Production leaves this empty
                so ``person_weights`` is loaded when ``person_source`` is
                ``coco``.
            project_root: Directory that relative weights paths start from.

        Raises:
            FileNotFoundError: ``model.weights`` does not exist. No other
                checkpoint is substituted.
            ValueError: ``person_source`` is not ``coco`` or ``ppe_model``.
        """
        model_cfg = settings["model"]
        self._root = project_root or PROJECT_ROOT
        self._person_name = str(settings["classes"]["person"])
        self._person_source = str(model_cfg["person_source"]).strip().lower()
        if self._person_source not in {"coco", "ppe_model"}:
            raise ValueError(
                f"model.person_source must be 'coco' or 'ppe_model', got {self._person_source!r}"
            )
        self._person_coco_class = int(model_cfg["person_coco_class"])
        self._person_conf = float(model_cfg["person_conf"])
        self._global_conf = float(model_cfg["confidence"])
        self._iou = float(model_cfg["iou"])
        self._imgsz = int(model_cfg["imgsz"])
        self._class_conf = {str(key): float(value) for key, value in model_cfg["class_confidence"].items()}
        self._device = resolve_device(str(model_cfg["device"]))
        weights = _weights_path(str(model_cfg["weights"]), self._root)
        if ppe_model is None and not weights.is_file():
            raise FileNotFoundError(
                f"PPE weights not found at {weights}. "
                "Copy the trained checkpoint to models/best.pt. There is no fallback model."
            )
        self._ppe = ppe_model if ppe_model is not None else self._load_yolo(weights)
        self._person = None
        if self._person_source == "coco":
            if person_model is not None:
                self._person = person_model
            else:
                self._person = self._load_yolo(str(model_cfg["person_weights"]))
        self._warmup(self._ppe)
        if self._person is not None:
            self._warmup(self._person)

    @property
    def device(self) -> str:
        """Device string actually passed to the models."""
        return self._device

    def set_thresholds(
        self,
        *,
        confidence: float,
        person_conf: float,
        class_confidence: Mapping[str, float],
    ) -> None:
        """Replace confidence cutoffs without reloading either model.

        Args:
            confidence: Global floor used when a class has no override.
            person_conf: Cutoff for the person class.
            class_confidence: Per-class cutoffs. Replaces the previous map.
        """
        self._global_conf = float(confidence)
        self._person_conf = float(person_conf)
        self._class_conf = {str(key): float(value) for key, value in class_confidence.items()}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people and PPE items in one BGR frame.

        Args:
            frame: Image array shaped ``(H, W, 3)``.

        Returns:
            Person boxes followed by PPE boxes. Empty when nothing qualifies.

        Raises:
            ValueError: ``frame`` is not a 3-channel image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")
        reject = {self._person_name} if self._person_source == "coco" else set()
        class_conf = dict(self._class_conf)
        if self._person_source == "ppe_model":
            class_conf[self._person_name] = self._person_conf
        ppe_boxes = self._predict(
            self._ppe,
            frame,
            confidence=self._predict_floor(),
            classes=self._ppe_class_ids(reject),
        )
        ppe = detections_from_boxes(
            _first_boxes(ppe_boxes),
            _names_of(self._ppe, ppe_boxes),
            class_confidence=class_conf,
            default_confidence=self._global_conf,
            reject_names=reject,
        )
        if self._person_source == "coco":
            assert self._person is not None
            person_boxes = self._predict(
                self._person,
                frame,
                confidence=self._person_conf,
                classes=[self._person_coco_class],
            )
            people = detections_from_boxes(
                _first_boxes(person_boxes),
                _names_of(self._person, person_boxes),
                class_confidence={self._person_name: self._person_conf},
                default_confidence=self._person_conf,
                reject_names=set(),
                allow_ids={self._person_coco_class},
                rename_ids={self._person_coco_class: self._person_name},
            )
            gear = ppe
        else:
            people = [det for det in ppe if det.class_name == self._person_name]
            gear = [det for det in ppe if det.class_name != self._person_name]
        return people + gear

    def _predict_floor(self) -> float:
        """Lowest cutoff, so a strict model threshold cannot drop gloves early."""
        values = [self._global_conf, *self._class_conf.values()]
        if self._person_source == "ppe_model":
            values.append(self._person_conf)
        return min(values)

    def _ppe_class_ids(self, reject: set[str]) -> list[int] | None:
        """Class indexes to request from the PPE model, or ``None`` when unknown."""
        names = getattr(self._ppe, "names", None)
        if not isinstance(names, Mapping) or not names:
            return None
        kept = [int(index) for index, label in names.items() if str(label) not in reject]
        return kept or None

    def _predict(
        self,
        model: Any,
        frame: np.ndarray,
        *,
        confidence: float,
        classes: list[int] | None,
    ) -> Any:
        """Run one model. ``classes`` is omitted when the model has no name map."""
        kwargs: dict[str, Any] = {
            "imgsz": self._imgsz,
            "conf": confidence,
            "iou": self._iou,
            "device": self._device,
            "verbose": False,
        }
        if classes is not None:
            kwargs["classes"] = classes
        return model.predict(frame, **kwargs)

    def _warmup(self, model: Any) -> None:
        """Run one blank frame so the first real frame does not pay load cost."""
        blank = np.zeros((16, 16, 3), dtype=np.uint8)
        self._predict(model, blank, confidence=self._predict_floor(), classes=None)

    @staticmethod
    def _load_yolo(weights: Path | str) -> Any:
        """Construct an ultralytics YOLO model."""
        from ultralytics import YOLO

        return YOLO(str(weights))


def _first_boxes(result: Any) -> Any:
    """Return the boxes of the first predict result."""
    if result is None:
        return None
    first = result[0] if isinstance(result, (list, tuple)) else result
    return getattr(first, "boxes", None)


def _names_of(model: Any, result: Any) -> Mapping[int, str]:
    """Prefer names stored on the result, then names stored on the model."""
    first = None
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
    names = getattr(first, "names", None)
    if isinstance(names, Mapping) and names:
        return {int(key): str(value) for key, value in names.items()}
    model_names = getattr(model, "names", {})
    if isinstance(model_names, Mapping):
        return {int(key): str(value) for key, value in model_names.items()}
    return {}
