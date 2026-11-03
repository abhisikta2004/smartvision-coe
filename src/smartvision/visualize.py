"""Draw zones, person boxes, and detected PPE on a frame.

Every person box is green. The color marks that a person was detected, not
whether their PPE is complete. Each detected PPE item is drawn in the
configured PPE color, which is red in the shipped settings. Missing PPE is
not drawn, because there is no box for it. When more than one person is in
the frame, each person box names that worker and the PPE on them, and each
PPE box names the worker it belongs to. Colors and line thickness come
from ``visualize`` in ``config/settings.yaml``. The input frame is not modified.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from smartvision.detector import Detection
from smartvision.zones import Zone, draw_zones


class FrameVisualizer:
    """Render pipeline state onto a copy of the input frame."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Store visual settings.

        Args:
            settings: Parsed ``config/settings.yaml``. The ``visualize``
                section supplies colors and thickness.
        """
        visual = settings["visualize"]
        self._colors = {
            str(name): (int(value[0]), int(value[1]), int(value[2]))
            for name, value in visual["colors"].items()
        }
        self._thickness = int(visual["thickness"])
        self._person_thickness = int(visual.get("person_thickness", self._thickness))
        self._ppe_thickness = int(visual.get("ppe_thickness", self._thickness))
        self._font_scale = float(visual["font_scale"])
        self._person_name = str(settings["classes"]["person"])
        labels = settings["classes"].get("display_names", {})
        self._labels = {str(name): str(label) for name, label in labels.items()}

    def draw(
        self,
        frame: np.ndarray,
        zones: Sequence[Zone],
        workers: Sequence[Any],
        detections: Sequence[Detection],
        show_zones: bool = True,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with overlays applied.

        Args:
            frame: BGR image shaped ``(H, W, 3)``.
            zones: Danger-zone polygons to fill and outline. Ignored when
                ``show_zones`` is false.
            workers: Objects with ``track_id``, ``bbox_xyxy``, and the PPE
                fields used for a multi-person caption.
            detections: Person and PPE boxes. Person boxes are drawn from
                ``workers`` in green. PPE boxes are drawn from
                ``detections``. When several workers are present, a PPE box
                assigned to one of them is labeled with that worker.
            show_zones: When false, the zone fill and outline are left off.
                Photos and recorded video use this. The live camera keeps
                the polygon visible.

        Returns:
            Annotated BGR image. The input array is left unchanged.
        """
        canvas = (
            draw_zones(frame, list(zones), self._colors["zone"], self._thickness)
            if show_zones
            else frame.copy()
        )
        several = len(workers) > 1
        claimed = _claimed_gear(workers) if several else set()
        for detection in detections:
            if detection.class_name == self._person_name:
                continue
            if (detection.class_name, detection.bbox_xyxy) in claimed:
                continue
            self._draw_box(
                canvas,
                detection.bbox_xyxy,
                self._colors["ppe"],
                self._label(detection.class_name),
                self._ppe_thickness,
            )
        if several:
            for worker in workers:
                for detection in getattr(worker, "gear", ()):
                    self._draw_box(
                        canvas,
                        detection.bbox_xyxy,
                        self._colors["ppe"],
                        f"Worker {worker.track_id} · {self._label(detection.class_name)}",
                        self._ppe_thickness,
                    )
        for worker in workers:
            label = (
                worker_caption(worker, self._labels)
                if several
                else f"Person {worker.track_id}"
            )
            self._draw_box(
                canvas,
                worker.bbox_xyxy,
                self._colors["ok"],
                label,
                self._person_thickness,
            )
        return canvas

    def _label(self, class_name: str) -> str:
        """Return the display name for a class."""
        return self._labels.get(class_name, class_name.replace("_", " ").title())

    def _draw_box(
        self,
        canvas: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        color: tuple[int, int, int],
        label: str,
        thickness: int | None = None,
    ) -> None:
        """Draw one rectangle and a label just above it.

        Person and PPE strokes are scaled with the frame width. Photos are
        drawn at full size and then shown at most 960 pixels wide, so a
        fixed stroke would vanish on a large image.
        """
        width = int(canvas.shape[1])
        base = self._thickness if thickness is None else int(thickness)
        stroke = max(base, int(round(base * width / 960)))
        x1, y1, x2, y2 = (int(round(value)) for value in bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, stroke)
        scale, weight = _text_size(self._font_scale, int(canvas.shape[1]))
        _draw_label(canvas, label, x1, y1, y2, scale, weight)


def worker_caption(worker: Any, labels: dict[str, str]) -> str:
    """Name one worker and the PPE that explains their box.

    Args:
        worker: Object with ``track_id``, ``worn_ppe``, and ``missing_ppe``.
        labels: Display names keyed by class name.

    Returns:
        A single line such as ``Worker 1 · Helmet · missing Safety Vest``.
    """
    worn = _ppe_names(getattr(worker, "worn_ppe", ()), labels)
    missing = _ppe_names(getattr(worker, "missing_ppe", ()), labels)
    parts = [f"Worker {worker.track_id}"]
    if worn:
        parts.append(worn)
    if missing:
        parts.append(f"missing {missing}")
    return " · ".join(parts)


def _ppe_names(names: Sequence[str], labels: dict[str, str]) -> str:
    """Join PPE display names in the order they were given."""
    return ", ".join(labels.get(name, name.replace("_", " ").title()) for name in names)


def _claimed_gear(workers: Sequence[Any]) -> set[tuple[str, tuple[float, float, float, float]]]:
    """Return the PPE boxes that already belong to a worker."""
    claimed: set[tuple[str, tuple[float, float, float, float]]] = set()
    for worker in workers:
        for detection in getattr(worker, "gear", ()):
            claimed.add((detection.class_name, detection.bbox_xyxy))
    return claimed


def _draw_label(
    canvas: np.ndarray,
    label: str,
    x1: int,
    y1: int,
    y2: int,
    scale: float,
    weight: int,
) -> None:
    """Draw one label on a dark chip so it stays readable over the photo."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, weight)
    height, width = canvas.shape[:2]
    top = y1 - text_h - baseline - 4
    if top < 0:
        top = min(y2 + 2, height - text_h - baseline - 2)
    top = max(0, top)
    left = min(max(0, x1), max(0, width - text_w - 2))
    cv2.rectangle(
        canvas,
        (left, top),
        (min(width - 1, left + text_w + 4), min(height - 1, top + text_h + baseline + 4)),
        (28, 25, 23),
        -1,
    )
    cv2.putText(
        canvas,
        label,
        (left + 2, top + text_h + 2),
        font,
        scale,
        (255, 255, 255),
        weight,
        cv2.LINE_AA,
    )


def _text_size(font_scale: float, width: int) -> tuple[float, int]:
    """Scale label type so it stays readable after the 960-pixel display shrink."""
    factor = max(1.0, width / 960.0)
    return font_scale * factor, max(1, int(round(factor)))


