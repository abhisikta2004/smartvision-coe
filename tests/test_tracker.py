"""Tests for stable IoU worker ids and track expiry."""

from __future__ import annotations

from smartvision.detector import Detection
from smartvision.tracker import IoUTracker, box_iou


def _person(bbox: tuple[float, float, float, float]) -> Detection:
    return Detection("person", 0.9, bbox)


def test_iou_is_one_for_the_same_box_and_zero_when_disjoint() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_matching_box_keeps_its_id_and_a_distant_box_gets_a_new_one(settings: dict) -> None:
    tracker = IoUTracker(settings)
    first = tracker.update(0, [_person((0, 0, 10, 20))])
    second = tracker.update(1, [_person((0, 0, 10, 20)), _person((100, 100, 110, 120))])
    assert first[0].track_id == 1
    ids = {track.track_id for track in second}
    assert ids == {1, 2}
    assert second[0].bbox_xyxy == (0, 0, 10, 20)


def test_track_expires_after_more_misses_than_the_configured_limit(settings: dict) -> None:
    settings["tracking"]["max_lost_frames"] = 1
    tracker = IoUTracker(settings)
    assert tracker.update(0, [_person((0, 0, 10, 10))])[0].track_id == 1
    assert tracker.update(1, []) == []
    assert tracker.update(2, [_person((0, 0, 10, 10))])[0].track_id == 1
    assert tracker.update(3, []) == []
    assert tracker.update(4, []) == []
    assert tracker.update(5, [_person((0, 0, 10, 10))])[0].track_id == 2


def test_low_overlap_does_not_reuse_an_id(settings: dict) -> None:
    settings["tracking"]["iou_threshold"] = 0.8
    tracker = IoUTracker(settings)
    tracker.update(0, [_person((0, 0, 10, 10))])
    tracks = tracker.update(1, [_person((8, 0, 18, 10))])
    assert tracks[0].track_id == 2
