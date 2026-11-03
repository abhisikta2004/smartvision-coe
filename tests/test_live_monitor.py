"""Tests for the live camera session. No camera and no model."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.live_server import create_app
from smartvision.detector import Detection
from smartvision.live_monitor import LiveMonitor
from smartvision.pipeline import SmartVisionPipeline
from smartvision.zones import Zone


class HoldingCapture:
    """Capture that keeps returning one frame until it is released."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False
        self._open = threading.Event()
        self._open.set()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.released:
            return False, None
        time.sleep(0.01)
        return True, self.frame.copy()

    def isOpened(self) -> bool:
        return not self.released

    def release(self) -> None:
        self.released = True


class ScriptedDetector:
    """Returns one fixed detection list."""

    def __init__(self, found: list[Detection]) -> None:
        self.found = found
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        self.calls += 1
        return list(self.found)


class Clock:
    """Manual clock for the safety interval."""

    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def _full_zone_person() -> Detection:
    return Detection("person", 0.9, (0.0, 0.0, 40.0, 40.0))


def _monitor(tmp_path: Path, settings: dict, frame: np.ndarray, clock: Clock | None = None) -> tuple[LiveMonitor, HoldingCapture, ScriptedDetector]:
    capture = HoldingCapture(frame)
    detector = ScriptedDetector([_full_zone_person()])

    def factory(zones: list[Zone]) -> SmartVisionPipeline:
        return SmartVisionPipeline(settings, zones, detector, project_root=tmp_path)

    monitor = LiveMonitor(
        settings,
        project_root=tmp_path,
        zones_path=tmp_path / "webcam_zone.yaml",
        open_capture=lambda _source: capture,
        pipeline_factory=factory,
        clock=clock,
    )
    return monitor, capture, detector


def test_save_zone_keeps_the_camera_open_and_starts_detection(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    monitor, capture, detector = _monitor(tmp_path, settings, frame)
    monitor.start("0")
    zone = monitor.save_zone("Machine 1", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], ["helmet", "vest"])
    assert zone.name == "Machine 1"
    assert capture.released is False
    assert monitor.snapshot()["monitoring"] is True
    assert (tmp_path / "webcam_zone.yaml").is_file()
    deadline = time.time() + 2.0
    while detector.calls < 1 and time.time() < deadline:
        time.sleep(0.02)
    assert detector.calls >= 1
    monitor.edit_zone()
    assert capture.released is False
    assert monitor.snapshot()["monitoring"] is False
    monitor.stop()
    assert capture.released is True


def test_status_waits_for_the_interval(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    clock = Clock()
    settings["alerts"]["live_status_seconds"] = 5.0
    monitor, _capture, _detector = _monitor(tmp_path, settings, frame, clock)
    monitor._capture = HoldingCapture(frame)
    monitor.save_zone("Machine 1", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], ["helmet", "vest"])
    monitor.publish_frame(frame)
    first = monitor.snapshot()["lines"]
    assert first
    assert "Machine 1" in first[0]["message"]
    clock.now += 4.0
    monitor.publish_frame(frame)
    assert monitor.snapshot()["lines"][0]["updated_at"] == first[0]["updated_at"]
    clock.now += 1.0
    monitor.publish_frame(frame)
    assert monitor.snapshot()["lines"][0]["updated_at"] != first[0]["updated_at"]


def test_bad_polygon_is_rejected(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    monitor, capture, _detector = _monitor(tmp_path, settings, frame)
    monitor._capture = capture
    try:
        monitor.save_zone("Machine 1", [(0.0, 0.0), (1.0, 0.0)], ["helmet"])
    except ValueError as exc:
        assert "3 points" in str(exc)
    else:
        raise AssertionError("expected a polygon error")
    assert capture.released is False


def test_api_save_does_not_release_the_camera(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    monitor, capture, _detector = _monitor(tmp_path, settings, frame)
    client = TestClient(create_app(monitor))
    opened = client.post("/api/camera", json={"index": 0})
    assert opened.status_code == 200
    saved = client.post(
        "/api/zone",
        json={"name": "Machine 1", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], "ppe": ["helmet", "vest"]},
    )
    assert saved.status_code == 200
    assert saved.json()["monitoring"] is True
    assert capture.released is False
    clicked = client.post(
        "/api/zone",
        json={
            "name": "Machine 1",
            "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 0.5, "y": 1.0}],
            "ppe": ["helmet", "vest"],
        },
    )
    assert clicked.status_code == 200
    assert capture.released is False
    page = client.get("/")
    assert page.status_code == 200
    assert "Draw polygon zone" in page.text
    monitor.stop()


def test_publish_frame_detects_without_opening_another_camera(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    monitor, capture, detector = _monitor(tmp_path, settings, frame)
    monitor._capture = capture
    monitor.save_zone("Machine 1", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], ["helmet", "vest"])
    assert detector.calls == 0
    monitor.publish_frame(frame)
    assert detector.calls == 1
    assert capture.released is False
