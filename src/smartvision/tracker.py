"""IoU worker tracking.

Person boxes are matched to existing workers so the same person keeps one
id across frames. The IoU threshold and how many missed frames a track
survives come from ``tracking`` in ``config/settings.yaml``. PPE boxes are
not tracked. This module does not open a camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from smartvision.detector import Detection
from smartvision.zones import Point, person_anchor


@dataclass
class Track:
    """One worker identity across frames.

    Attributes:
        track_id: Stable id. Ids are not reused.
        bbox_xyxy: Latest person box as ``(x1, y1, x2, y2)`` in pixels.
        anchor: Bottom-center of ``bbox_xyxy``.
        last_seen_frame: Frame index of the last match.
        missed: Consecutive updates without a match.
    """

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    anchor: Point | None = None
    last_seen_frame: int = 0
    missed: int = 0


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return the intersection-over-union of two boxes.

    Args:
        first: ``(x1, y1, x2, y2)``.
        second: ``(x1, y1, x2, y2)``.

    Returns:
        IoU in ``[0, 1]``. Zero when either box has no area or they miss.
    """
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    left = max(min(ax1, ax2), min(bx1, bx2))
    right = min(max(ax1, ax2), max(bx1, bx2))
    top = max(min(ay1, ay2), min(by1, by2))
    bottom = min(max(ay1, ay2), max(by1, by2))
    width = right - left
    height = bottom - top
    intersection = width * height if width > 0 and height > 0 else 0.0
    area_a = max(0.0, abs(ax2 - ax1)) * max(0.0, abs(ay2 - ay1))
    area_b = max(0.0, abs(bx2 - bx1)) * max(0.0, abs(by2 - by1))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def greedy_matches(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    threshold: float,
) -> list[tuple[int, int]]:
    """Match detections to tracks by descending IoU.

    Args:
        tracks: Active tracks.
        detections: Person detections for this frame.
        threshold: Minimum IoU that counts as the same worker.

    Returns:
        Pairs of ``(track_index, detection_index)``. Each index is used
        at most once.
    """
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for det_index, detection in enumerate(detections):
            score = box_iou(track.bbox_xyxy, detection.bbox_xyxy)
            if score >= threshold:
                pairs.append((score, track_index, det_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, track_index, det_index in pairs:
        if track_index in used_tracks or det_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(det_index)
        matches.append((track_index, det_index))
    return matches


class IoUTracker:
    """Associate person boxes by IoU and keep stable worker ids."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Store tracking settings.

        Args:
            settings: Parsed ``config/settings.yaml``. The ``tracking``
                section supplies the IoU threshold and the expiry.
        """
        tracking = settings["tracking"]
        self._threshold = float(tracking["iou_threshold"])
        self._max_lost = int(tracking["max_lost_frames"])
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, frame_index: int, persons: list[Detection]) -> list[Track]:
        """Match person detections to tracks and return the visible set.

        A track that has already missed more than
        ``tracking.max_lost_frames`` updates is dropped before matching,
        so a later detection receives a new id. Ids already issued are
        never reused.

        Args:
            frame_index: Monotonic index of the frame being processed.
            persons: Person detections for this frame.

        Returns:
            Tracks matched on this frame, including brand-new ids.
        """
        self._tracks = [track for track in self._tracks if track.missed <= self._max_lost]
        matches = greedy_matches(self._tracks, persons, self._threshold)
        matched_tracks = {track_index for track_index, _det_index in matches}
        matched_detections = {det_index for _track_index, det_index in matches}
        visible: list[Track] = []
        for track_index, det_index in matches:
            track = self._tracks[track_index]
            detection = persons[det_index]
            track.bbox_xyxy = detection.bbox_xyxy
            track.anchor = person_anchor(detection.bbox_xyxy)
            track.last_seen_frame = frame_index
            track.missed = 0
            visible.append(track)
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.missed += 1
        for det_index, detection in enumerate(persons):
            if det_index in matched_detections:
                continue
            track = Track(
                track_id=self._next_id,
                bbox_xyxy=detection.bbox_xyxy,
                anchor=person_anchor(detection.bbox_xyxy),
                last_seen_frame=frame_index,
                missed=0,
            )
            self._next_id += 1
            self._tracks.append(track)
            visible.append(track)
        return visible
