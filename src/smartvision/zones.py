"""Person anchor and danger-zone polygons.

A person is localized at the bottom-center of their box. That pixel is
tested against the normalized polygons in ``config/zones.yaml`` with
``cv2.pointPolygonTest``. When several zones contain the point, the highest
``risk_level`` wins. This module accepts numpy frames and does not open a
camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Point:
    """A pixel or normalized image point.

    Attributes:
        x: Horizontal coordinate.
        y: Vertical coordinate.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Zone:
    """One danger zone and the PPE it requires.

    Attributes:
        zone_id: Stable id from ``config/zones.yaml``.
        name: Label drawn on the overlay.
        polygon: Vertices in order, normalized to ``[0, 1]``. The polygon
            is closed implicitly.
        required_ppe: Class names that must be worn inside this zone.
        risk_level: Higher values win when polygons overlap.
    """

    zone_id: str
    name: str
    polygon: tuple[Point, ...]
    required_ppe: tuple[str, ...]
    risk_level: int = 0


def person_anchor(bbox_xyxy: tuple[float, float, float, float]) -> Point:
    """Return the bottom-center of a person box.

    Args:
        bbox_xyxy: Person box as ``(x1, y1, x2, y2)`` in pixels.

    Returns:
        The point ``((x1 + x2) / 2, y2)``.
    """
    x1, _y1, x2, y2 = bbox_xyxy
    return Point((float(x1) + float(x2)) / 2.0, float(y2))


def bottom_center(bbox_xyxy: tuple[float, float, float, float]) -> Point:
    """Return the bottom-center of a person box.

    Args:
        bbox_xyxy: Person box as ``(x1, y1, x2, y2)`` in pixels.

    Returns:
        The same point as :func:`person_anchor`.
    """
    return person_anchor(bbox_xyxy)


def load_zones(path: str | Path) -> list[Zone]:
    """Load danger zones from a YAML file.

    Args:
        path: Path to ``config/zones.yaml`` or a file written by
            :func:`save_zones`.

    Returns:
        Zones in file order.

    Raises:
        ValueError: The file has no ``zones`` list, or a zone has fewer
            than three polygon points.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("zones"), list):
        raise ValueError(f"{path} must contain a zones list")
    zones: list[Zone] = []
    for entry in payload["zones"]:
        zones.append(_zone_from_mapping(entry))
    return zones


def save_zones(path: str | Path, zones: list[Zone]) -> None:
    """Write zones in the dashboard editor's YAML shape.

    Args:
        path: Destination file. Parent directories are created.
        zones: Zones to store. Polygons stay normalized.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"zones": [_zone_to_mapping(zone) for zone in zones]}
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def box_overlaps_zone(
    bbox_xyxy: tuple[float, float, float, float],
    zone: Zone,
    image_size: tuple[int, int],
) -> bool:
    """Report whether any part of a box meets a zone polygon.

    A person counts as inside when a corner lies in the polygon, a polygon
    vertex lies in the box, or an edge of the box crosses an edge of the
    polygon. The whole body does not have to enter.

    Args:
        bbox_xyxy: ``(x1, y1, x2, y2)`` in pixels.
        zone: Zone whose vertices are normalized.
        image_size: ``(width, height)`` used to scale the polygon.

    Returns:
        ``True`` when the box and the polygon share any area or boundary.
    """
    width, height = image_size
    contour = polygon_pixels(zone, width, height)
    if contour.shape[0] < 3:
        return False
    polygon = [(float(point[0][0]), float(point[0][1])) for point in contour]
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    for corner in corners:
        if cv2.pointPolygonTest(contour, corner, False) >= 0:
            return True
    for vertex in polygon:
        if left <= vertex[0] <= right and top <= vertex[1] <= bottom:
            return True
    polygon_edges = tuple(zip(polygon, polygon[1:] + polygon[:1]))
    box_edges = tuple(zip(corners, corners[1:] + corners[:1]))
    return any(
        _segments_intersect(start, end, other_start, other_end)
        for start, end in box_edges
        for other_start, other_end in polygon_edges
    )


def find_zone_for_box(
    bbox_xyxy: tuple[float, float, float, float],
    zones: list[Zone],
    image_size: tuple[int, int],
) -> Zone | None:
    """Return the highest-risk zone that overlaps a person box.

    Args:
        bbox_xyxy: Person box as ``(x1, y1, x2, y2)`` in pixels.
        zones: Candidates.
        image_size: ``(width, height)`` of the frame.

    Returns:
        The winning zone, or ``None`` when the box misses every polygon.
        Equal risk keeps the earlier zone in ``zones``.
    """
    containing = [zone for zone in zones if box_overlaps_zone(bbox_xyxy, zone, image_size)]
    if not containing:
        return None
    return max(enumerate(containing), key=lambda item: (item[1].risk_level, -item[0]))[1]


def find_zone(
    point: Point,
    zones: list[Zone],
    image_size: tuple[int, int],
) -> Zone | None:
    """Return the containing zone with the highest risk.

    Args:
        point: Pixel coordinate, normally a person anchor.
        zones: Candidates from :func:`load_zones`.
        image_size: ``(width, height)`` of the frame. Normalized polygon
            vertices are scaled by these values.

    Returns:
        The winning zone, or ``None`` when the point is outside all of them.
        Equal risk keeps the earlier zone in ``zones``.
    """
    containing = zones_containing(point, zones, image_size)
    if not containing:
        return None
    return max(enumerate(containing), key=lambda item: (item[1].risk_level, -item[0]))[1]


def zones_containing(
    point: Point,
    zones: list[Zone],
    image_size: tuple[int, int],
) -> list[Zone]:
    """Return every zone whose polygon contains ``point``.

    Args:
        point: Pixel coordinate.
        zones: Candidates.
        image_size: ``(width, height)`` used to scale normalized vertices.

    Returns:
        Matching zones in input order. A point on the boundary counts as
        inside.
    """
    width, height = image_size
    matched: list[Zone] = []
    for zone in zones:
        contour = polygon_pixels(zone, width, height)
        if contour.shape[0] < 3:
            continue
        test = cv2.pointPolygonTest(contour, (float(point.x), float(point.y)), False)
        if test >= 0:
            matched.append(zone)
    return matched


def polygon_pixels(zone: Zone, width: int, height: int) -> np.ndarray:
    """Scale a normalized polygon into pixel coordinates.

    Args:
        zone: Zone whose vertices are in ``[0, 1]``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Array shaped ``(N, 1, 2)`` for OpenCV polygon calls.
    """
    points = [
        [point.x * width, point.y * height]
        for point in zone.polygon
    ]
    return np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))


def draw_zones(
    frame: np.ndarray,
    zones: list[Zone],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw zone polygons on a copy of ``frame``.

    Args:
        frame: BGR image shaped ``(H, W, 3)``.
        zones: Polygons to fill lightly and outline.
        color: BGR outline color.
        thickness: Outline width in pixels.

    Returns:
        Annotated BGR image. The input array is left unchanged.
    """
    canvas = frame.copy()
    height, width = frame.shape[:2]
    overlay = canvas.copy()
    contours = []
    for zone in zones:
        contour = np.round(polygon_pixels(zone, width, height)).astype(np.int32)
        if contour.shape[0] < 3:
            continue
        contours.append(contour)
        origin = contour[0, 0]
        cv2.putText(
            canvas,
            zone.name,
            (int(origin[0]), max(12, int(origin[1]) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if contours:
        cv2.fillPoly(overlay, contours, color)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, contours, True, color, thickness)
    return canvas


class ZoneChecker:
    """Test person anchors against a fixed zone list."""

    def __init__(self, zones: list[Zone]) -> None:
        """Store zones.

        Args:
            zones: Zones already loaded from config.
        """
        self._zones = list(zones)

    @property
    def zones(self) -> list[Zone]:
        """Zones in config order."""
        return list(self._zones)

    def zones_containing(self, point: Point, image_size: tuple[int, int]) -> list[Zone]:
        """Return every stored zone that contains ``point``.

        Args:
            point: Person anchor in pixels.
            image_size: ``(width, height)`` of the frame.

        Returns:
            Matching zones. Empty when the person is outside every zone.
        """
        return zones_containing(point, self._zones, image_size)

    def find_zone(self, point: Point, image_size: tuple[int, int]) -> Zone | None:
        """Return the highest-risk stored zone containing ``point``.

        Args:
            point: Person anchor in pixels.
            image_size: ``(width, height)`` of the frame.

        Returns:
            The winning zone, or ``None`` when the point is outside.
        """
        return find_zone(point, self._zones, image_size)


def _zone_from_mapping(entry: object) -> Zone:
    """Build one zone from a YAML mapping."""
    if not isinstance(entry, dict):
        raise ValueError("each zone must be a mapping")
    name = str(entry["name"])
    zone_id = str(entry.get("id") or _slug(name))
    raw_points = entry.get("polygon") or []
    if len(raw_points) < 3:
        raise ValueError(f"zone {name!r} needs at least three polygon points")
    polygon = tuple(Point(float(point[0]), float(point[1])) for point in raw_points)
    required = tuple(str(item) for item in entry.get("required_ppe") or [])
    return Zone(
        zone_id=zone_id,
        name=name,
        polygon=polygon,
        required_ppe=required,
        risk_level=int(entry.get("risk_level", 0)),
    )


def _zone_to_mapping(zone: Zone) -> dict[str, object]:
    """Serialize one zone for :func:`save_zones`."""
    return {
        "id": zone.zone_id,
        "name": zone.name,
        "risk_level": zone.risk_level,
        "required_ppe": list(zone.required_ppe),
        "polygon": [[point.x, point.y] for point in zone.polygon],
    }


def _segments_intersect(
    start: tuple[float, float],
    end: tuple[float, float],
    other_start: tuple[float, float],
    other_end: tuple[float, float],
) -> bool:
    """Report whether two segments touch, including at an endpoint."""

    def orient(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])

    def on_segment(
        first: tuple[float, float],
        middle: tuple[float, float],
        last: tuple[float, float],
    ) -> bool:
        return (
            min(first[0], last[0]) - 1e-6 <= middle[0] <= max(first[0], last[0]) + 1e-6
            and min(first[1], last[1]) - 1e-6 <= middle[1] <= max(first[1], last[1]) + 1e-6
        )

    first = orient(start, end, other_start)
    second = orient(start, end, other_end)
    third = orient(other_start, other_end, start)
    fourth = orient(other_start, other_end, end)
    if abs(first) < 1e-6 and on_segment(start, other_start, end):
        return True
    if abs(second) < 1e-6 and on_segment(start, other_end, end):
        return True
    if abs(third) < 1e-6 and on_segment(other_start, start, other_end):
        return True
    if abs(fourth) < 1e-6 and on_segment(other_start, end, other_end):
        return True
    return (first > 0) != (second > 0) and (third > 0) != (fourth > 0)


def _slug(name: str) -> str:
    """Turn a zone name into a stable id."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "zone"
