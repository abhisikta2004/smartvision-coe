"""Assign PPE boxes to people and compare them with the required set.

A PPE detection counts for a person when its center lies inside that
person's box and inside the body band configured for its class: helmet and
goggles in the upper third, vest in the middle, gloves from the middle
down, boots in the bottom quarter. If the center falls in more than one
person, the person with the larger box overlap wins. The required set is
the zone's ``required_ppe`` inside a zone, and
``compliance.default_required_ppe`` outside. This module does not keep
time and does not raise alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from smartvision.detector import Detection
from smartvision.zones import Zone


@dataclass(frozen=True)
class ComplianceResult:
    """PPE outcome for one person on one frame.

    Attributes:
        worn: Required classes detected on the person, in ``classes.ppe``
            order. A class that was not selected is omitted.
        missing: Required classes that were not assigned and could be checked.
        is_compliant: ``True`` when ``missing`` is empty. An unverified item
            does not by itself make the worker non-compliant.
        required_ppe: The set that was checked.
        uncertain: Required classes whose body band is too cropped to confirm.
        assigned: PPE detections kept for this person. Empty when the
            caller does not need the boxes.
    """

    worn: tuple[str, ...]
    missing: tuple[str, ...]
    is_compliant: bool
    required_ppe: tuple[str, ...]
    uncertain: tuple[str, ...] = ()
    assigned: tuple[Detection, ...] = ()


def box_center(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the center of a box.

    Args:
        bbox_xyxy: ``(x1, y1, x2, y2)`` in pixels.

    Returns:
        ``((x1 + x2) / 2, (y1 + y2) / 2)``.
    """
    x1, y1, x2, y2 = bbox_xyxy
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def point_in_box(
    point: tuple[float, float],
    bbox_xyxy: tuple[float, float, float, float],
) -> bool:
    """Report whether ``point`` lies inside the box, edges included.

    Args:
        point: ``(x, y)`` in pixels.
        bbox_xyxy: ``(x1, y1, x2, y2)`` in pixels.

    Returns:
        ``True`` when the point is inside or on the edge.
    """
    x1, y1, x2, y2 = bbox_xyxy
    x_pos, y_pos = point
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return left <= x_pos <= right and top <= y_pos <= bottom


def intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return the overlap area of two boxes.

    Args:
        first: ``(x1, y1, x2, y2)``.
        second: ``(x1, y1, x2, y2)``.

    Returns:
        Overlap area in pixels squared, or ``0`` when the boxes miss.
    """
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    left = max(min(ax1, ax2), min(bx1, bx2))
    right = min(max(ax1, ax2), max(bx1, bx2))
    top = max(min(ay1, ay2), min(by1, by2))
    bottom = min(max(ay1, ay2), max(by1, by2))
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return 0.0
    return float(width * height)


def vertical_band(
    bbox_xyxy: tuple[float, float, float, float],
    y_start: float,
    y_end: float,
) -> tuple[float, float, float, float]:
    """Slice a person box into a vertical fraction of its height.

    Args:
        bbox_xyxy: Person box as ``(x1, y1, x2, y2)``.
        y_start: Start of the band, 0 at the top of the box.
        y_end: End of the band, 1 at the bottom of the box.

    Returns:
        The band as ``(x1, y1, x2, y2)``. A zero-height box is returned
        unchanged.
    """
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    height = y2 - y1
    if height <= 0:
        return (x1, y1, x2, y2)
    return (x1, y1 + y_start * height, x2, y1 + y_end * height)


def assign_ppe(
    people: Sequence[Detection],
    ppe_detections: Sequence[Detection],
    regions: Mapping[str, tuple[float, float]],
) -> list[list[Detection]]:
    """Assign each PPE box to at most one person.

    The center must lie inside the person's box and inside the configured
    body band for that class. A class with no band uses the full person
    box. When several people qualify, the larger intersection area wins,
    and an equal area keeps the earlier person.

    Args:
        people: Person detections.
        ppe_detections: PPE detections for the same frame.
        regions: Map of class name to ``(y_start, y_end)`` fractions.

    Returns:
        One list of PPE detections per person, in ``people`` order.
    """
    assigned: list[list[Detection]] = [[] for _ in people]
    for detection in ppe_detections:
        center = box_center(detection.bbox_xyxy)
        candidates: list[tuple[float, int]] = []
        for index, person in enumerate(people):
            if not point_in_box(center, person.bbox_xyxy):
                continue
            band = regions.get(detection.class_name)
            target = person.bbox_xyxy if band is None else vertical_band(person.bbox_xyxy, band[0], band[1])
            if not point_in_box(center, target):
                continue
            area = intersection_area(detection.bbox_xyxy, person.bbox_xyxy)
            candidates.append((area, index))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], item[1]))
        assigned[candidates[0][1]].append(detection)
    return assigned


def visible_fraction(
    bbox_xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> float:
    """Return how much of a box lies inside the frame.

    Args:
        bbox_xyxy: ``(x1, y1, x2, y2)`` in pixels. It may extend outside.
        image_size: ``(width, height)`` of the frame.

    Returns:
        Visible area divided by the box area. ``1`` when the box is fully
        inside. ``0`` when it misses the frame or has no area.
    """
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    area = max(0.0, right - left) * max(0.0, bottom - top)
    if area <= 0:
        return 0.0
    clipped_left = max(0.0, left)
    clipped_right = min(float(width), right)
    clipped_top = max(0.0, top)
    clipped_bottom = min(float(height), bottom)
    visible = max(0.0, clipped_right - clipped_left) * max(0.0, clipped_bottom - clipped_top)
    return float(visible / area)


def classes_on_person(
    person_box: tuple[float, float, float, float],
    detections: Sequence[Detection],
    classes: Sequence[str],
) -> tuple[str, ...]:
    """Return required classes whose boxes overlap one person.

    The body-band check can reject a vest that is still drawn on the
    worker. This list is what the table should show as detected.

    Args:
        person_box: Person box as ``(x1, y1, x2, y2)`` in pixels.
        detections: PPE boxes from the same frame.
        classes: Required class names, in display order.

    Returns:
        Required classes that overlap the person, in ``classes`` order.
    """
    present = {
        detection.class_name
        for detection in detections
        if intersection_area(detection.bbox_xyxy, person_box) > 0
    }
    return tuple(name for name in classes if name in present)


def reconcile_worn(
    required: Sequence[str],
    worn: Sequence[str],
    missing: Sequence[str],
    seen: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], bool, tuple[str, ...]]:
    """Drop gear that overlaps the person from the missing list.

    Args:
        required: Checklist for this person, in display order.
        worn: Classes accepted by the body-band check.
        missing: Required classes the band check did not confirm.
        seen: Required classes whose boxes overlap the person.

    Returns:
        Worn classes, the remaining missing classes, whether nothing
        required is still missing, and the seen classes in checklist order.
    """
    seen_set = set(seen)
    worn_set = set(worn) | seen_set
    worn_out = tuple(name for name in required if name in worn_set)
    missing_out = tuple(name for name in missing if name not in seen_set)
    seen_out = tuple(name for name in required if name in seen_set)
    return worn_out, missing_out, not missing_out, seen_out


def clipped_classes(
    bbox_xyxy: tuple[float, float, float, float],
    regions: Mapping[str, tuple[float, float]],
    image_size: tuple[int, int],
    min_visible: float,
) -> tuple[str, ...]:
    """Return PPE classes whose body band is mostly outside the frame.

    Args:
        bbox_xyxy: Person box in pixels.
        regions: Map of class name to ``(y_start, y_end)`` fractions.
        image_size: ``(width, height)`` of the frame.
        min_visible: Fraction of the band that must be on screen before a
            missing detection is treated as confirmed.

    Returns:
        Class names in ``regions`` order. A class is included when less
        than ``min_visible`` of its band is inside the frame.
    """
    clipped: list[str] = []
    for name, (y_start, y_end) in regions.items():
        band = vertical_band(bbox_xyxy, y_start, y_end)
        if visible_fraction(band, image_size) < min_visible:
            clipped.append(str(name))
    return tuple(clipped)


def result_for(
    assigned: Sequence[Detection],
    required: Sequence[str],
    ppe_order: Sequence[str],
    uncertain: Sequence[str] = (),
) -> ComplianceResult:
    """Compare assigned PPE classes with the required set.

    Args:
        assigned: PPE detections kept for one person.
        required: Class names that must be present.
        ppe_order: Canonical class order used for ``worn``.
        uncertain: Required classes that cannot be confirmed because the
            matching body band is cropped. A class that was detected is
            worn, not uncertain.

    Returns:
        Detected required classes, confirmed missing classes, unverified
        classes, and the compliance flag. Classes outside ``required`` are
        left out of every list. ``assigned`` keeps the boxes that were
        matched to the person.
    """
    present = {detection.class_name for detection in assigned}
    required_tuple = tuple(dict.fromkeys(str(name) for name in required))
    required_set = set(required_tuple)
    uncertain_out = tuple(name for name in required_tuple if name in set(uncertain) and name not in present)
    uncertain_set = set(uncertain_out)
    missing = tuple(name for name in required_tuple if name not in present and name not in uncertain_set)
    worn_known = tuple(name for name in ppe_order if name in present and name in required_set)
    worn_rest = tuple(name for name in required_tuple if name in present and name not in set(worn_known))
    worn = worn_known + worn_rest
    return ComplianceResult(
        worn=worn,
        missing=missing,
        is_compliant=not missing,
        required_ppe=required_tuple,
        uncertain=uncertain_out,
        assigned=tuple(assigned),
    )


class ComplianceEngine:
    """Apply configured body regions and the required-PPE list."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Store compliance settings.

        Args:
            settings: Parsed ``config/settings.yaml``. ``compliance`` and
                ``classes`` are read.
        """
        compliance = settings["compliance"]
        self._default = tuple(str(name) for name in compliance["default_required_ppe"])
        self._regions = {
            str(name): (float(band["y_start"]), float(band["y_end"]))
            for name, band in compliance["body_regions"].items()
        }
        self._ppe_order = tuple(str(name) for name in settings["classes"]["ppe"])
        self._min_visible = float(compliance.get("min_band_visible", 0.5))

    def required_for(self, zone: Zone | None) -> tuple[str, ...]:
        """Return the PPE list that applies to one person.

        Args:
            zone: Zone that contains the person, or ``None`` when they are
                outside every zone.

        Returns:
            The zone list, or the default list from settings when ``zone``
            is ``None``.
        """
        if zone is None:
            return self._default
        return tuple(zone.required_ppe)

    def evaluate(
        self,
        people: Sequence[Detection],
        ppe_detections: Sequence[Detection],
        required: Sequence[Sequence[str]],
        image_size: tuple[int, int] | None = None,
    ) -> list[ComplianceResult]:
        """Evaluate every person on one frame.

        Args:
            people: Person detections.
            ppe_detections: PPE detections from the same frame.
            required: One required-class sequence per person.
            image_size: ``(width, height)`` of the frame. When set, a
                required class whose body band is mostly off screen is
                marked uncertain instead of missing.

        Returns:
            One result per person, in the same order.

        Raises:
            ValueError: ``required`` does not line up with ``people``.
        """
        if len(required) != len(people):
            raise ValueError("required PPE lists must match the person count")
        assigned = assign_ppe(people, ppe_detections, self._regions)
        results: list[ComplianceResult] = []
        for person, items, req in zip(people, assigned, required):
            clipped: tuple[str, ...] = ()
            if image_size is not None:
                clipped = clipped_classes(person.bbox_xyxy, self._regions, image_size, self._min_visible)
            results.append(result_for(items, req, self._ppe_order, clipped))
        return results
