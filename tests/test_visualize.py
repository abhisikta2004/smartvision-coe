"""Tests for status colors and labels. No camera and no model."""

from __future__ import annotations

import cv2
import numpy as np

from smartvision.detector import Detection
from smartvision.pipeline import Worker, WorkerStatus
from smartvision.visualize import FrameVisualizer, worker_caption
from smartvision.zones import Point, Zone


def _worker(status: WorkerStatus, missing: tuple[str, ...] = ()) -> Worker:
    return Worker(
        track_id=7,
        bbox_xyxy=(10.0, 10.0, 30.0, 40.0),
        zone=None,
        missing_ppe=missing,
        worn_ppe=(),
        status=status,
        is_compliant=status is WorkerStatus.OK,
    )


def test_draw_keeps_people_green_and_ppe_in_its_own_color(frame: np.ndarray, settings: dict) -> None:
    settings["visualize"]["colors"]["ppe"] = [0, 0, 220]
    visualizer = FrameVisualizer(settings)
    zone = Zone(
        zone_id="welding_zone",
        name="Welding Zone",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)),
        required_ppe=("helmet",),
        risk_level=1,
    )
    original = frame.copy()
    ok = visualizer.draw(frame, [zone], [_worker(WorkerStatus.OK)], [])
    warning = visualizer.draw(
        frame,
        [],
        [_worker(WorkerStatus.WARNING, ("helmet",))],
        [Detection("helmet", 0.5, (12.0, 12.0, 18.0, 18.0))],
    )
    critical = visualizer.draw(frame, [], [_worker(WorkerStatus.CRITICAL, ("vest",))], [])
    assert np.array_equal(frame, original)
    assert tuple(int(value) for value in ok[10, 10]) == (0, 180, 0)
    assert tuple(int(value) for value in warning[10, 10]) == (0, 180, 0)
    assert tuple(int(value) for value in critical[10, 10]) == (0, 180, 0)
    assert tuple(int(value) for value in warning[12, 15]) == (0, 0, 220)
    assert ok.sum() > 0


def test_image_and_video_leave_the_zone_undrawn(frame: np.ndarray, settings: dict) -> None:
    """A full-frame zone stays invisible when the overlay is asked to hide it."""
    visualizer = FrameVisualizer(settings)
    zone = Zone(
        zone_id="work_area",
        name="Work area",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)),
        required_ppe=("helmet",),
        risk_level=1,
    )
    hidden = visualizer.draw(frame, [zone], [], [], show_zones=False)
    shown = visualizer.draw(frame, [zone], [], [], show_zones=True)
    assert np.array_equal(hidden, frame)
    assert shown.sum() > frame.sum()


def test_ppe_border_is_thicker_than_the_person_box(frame: np.ndarray, settings: dict) -> None:
    """The PPE stroke reaches farther inside the box than the person stroke."""
    settings["visualize"]["ppe_thickness"] = 4
    settings["visualize"]["colors"]["ppe"] = [0, 0, 80]
    visualizer = FrameVisualizer(settings)
    drawn = visualizer.draw(
        frame,
        [],
        [_worker(WorkerStatus.OK)],
        [Detection("helmet", 0.9, (12.0, 12.0, 28.0, 28.0))],
    )
    assert tuple(int(value) for value in drawn[14, 20]) == (0, 0, 80)
    assert tuple(int(value) for value in drawn[20, 20]) == (0, 0, 0)


def test_ppe_border_stays_thick_after_the_display_resize() -> None:
    """A large photo keeps a visible PPE stroke once it is shown at 960 pixels."""
    settings = {
        "visualize": {
            "colors": {
                "ok": [0, 180, 0],
                "warning": [0, 165, 255],
                "critical": [0, 0, 220],
                "ppe": [0, 0, 80],
                "zone": [255, 128, 0],
            },
            "thickness": 2,
            "ppe_thickness": 4,
            "font_scale": 0.5,
        },
        "classes": {"person": "person", "display_names": {}},
    }
    frame = np.zeros((1200, 1920, 3), dtype=np.uint8)
    drawn = FrameVisualizer(settings).draw(
        frame,
        [],
        [],
        [Detection("helmet", 0.9, (400.0, 400.0, 700.0, 700.0))],
    )
    shown = cv2.resize(drawn, (960, 600))
    assert int(shown[200, 219, 2]) > 40


def test_worker_border_stays_visible_after_the_display_resize() -> None:
    """A worker box on a wide photo stays thick once the picture is shown at 960 pixels."""
    settings = {
        "visualize": {
            "colors": {
                "ok": [0, 180, 0],
                "warning": [0, 165, 255],
                "critical": [0, 0, 220],
                "ppe": [0, 0, 80],
                "zone": [255, 128, 0],
            },
            "thickness": 2,
            "person_thickness": 6,
            "ppe_thickness": 4,
            "font_scale": 0.5,
        },
        "classes": {"person": "person", "display_names": {}},
    }
    frame = np.zeros((1200, 1920, 3), dtype=np.uint8)
    worker = Worker(
        track_id=1,
        bbox_xyxy=(400.0, 400.0, 900.0, 900.0),
        zone=None,
        missing_ppe=(),
        worn_ppe=(),
        status=WorkerStatus.OK,
        is_compliant=True,
    )
    drawn = FrameVisualizer(settings).draw(frame, [], [worker], [])
    shown = cv2.resize(drawn, (960, 600))
    assert int(shown[300, 200, 1]) > 40


def test_worker_caption_names_the_person_and_their_ppe() -> None:
    """Each worker's line lists who they are and the gear that is missing."""
    labels = {"helmet": "Helmet", "vest": "Safety Vest"}
    first = Worker(
        track_id=1,
        bbox_xyxy=(10.0, 10.0, 30.0, 40.0),
        zone=None,
        missing_ppe=("vest",),
        worn_ppe=("helmet",),
        status=WorkerStatus.CRITICAL,
        is_compliant=False,
    )
    second = Worker(
        track_id=2,
        bbox_xyxy=(40.0, 10.0, 60.0, 40.0),
        zone=None,
        missing_ppe=("helmet", "vest"),
        worn_ppe=(),
        status=WorkerStatus.CRITICAL,
        is_compliant=False,
    )
    assert worker_caption(first, labels) == "Worker 1 · Helmet · missing Safety Vest"
    assert worker_caption(second, labels) == "Worker 2 · missing Helmet, Safety Vest"


def test_several_workers_get_a_box_for_their_own_ppe(frame: np.ndarray, settings: dict, monkeypatch) -> None:
    """Two people keep separate person boxes and their PPE boxes name the worker."""
    labels: list[str] = []
    real = cv2.putText

    def capture(image, text, *args, **kwargs):
        labels.append(text)
        return real(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    settings["visualize"]["colors"]["ppe"] = [0, 0, 90]
    settings["visualize"]["ppe_thickness"] = 4
    settings["classes"]["display_names"] = {"helmet": "Helmet", "vest": "Safety Vest"}
    helmet = Detection("helmet", 0.9, (12.0, 12.0, 20.0, 20.0))
    vest = Detection("vest", 0.9, (42.0, 18.0, 55.0, 30.0))
    first = Worker(
        track_id=1,
        bbox_xyxy=(10.0, 10.0, 30.0, 40.0),
        zone=None,
        missing_ppe=("vest",),
        worn_ppe=("helmet",),
        status=WorkerStatus.CRITICAL,
        is_compliant=False,
        gear=(helmet,),
    )
    second = Worker(
        track_id=2,
        bbox_xyxy=(40.0, 10.0, 60.0, 40.0),
        zone=None,
        missing_ppe=("helmet",),
        worn_ppe=("vest",),
        status=WorkerStatus.CRITICAL,
        is_compliant=False,
        gear=(vest,),
    )
    drawn = FrameVisualizer(settings).draw(frame, [], [first, second], [helmet, vest])
    assert tuple(int(value) for value in drawn[12, 16]) == (0, 0, 90)
    assert tuple(int(value) for value in drawn[18, 48]) == (0, 0, 90)
    assert tuple(int(value) for value in drawn[10, 10]) == (0, 180, 0)
    assert tuple(int(value) for value in drawn[10, 40]) == (0, 180, 0)
    assert "Worker 1 · Helmet" in labels
    assert "Worker 2 · Safety Vest" in labels
    assert any(text.startswith("Worker 1 ·") and "missing" in text for text in labels)
    assert any(text.startswith("Worker 2 ·") and "missing" in text for text in labels)
