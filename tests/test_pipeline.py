"""Pipeline tests with a scripted detector. No model and no camera."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from smartvision.detector import Detection
from smartvision.pipeline import SmartVisionPipeline, is_live_source, open_video_capture
from smartvision.zones import Point, Zone


class ScriptedDetector:
    """Returns the next scripted detection list on every call."""

    def __init__(self, script: list[list[Detection]]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        found = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return found


class FakeCapture:
    """In-memory stand-in for cv2.VideoCapture."""

    def __init__(self, frames: list[np.ndarray], fps: float = 10.0) -> None:
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, _prop: int) -> float:
        return self.fps

    def release(self) -> None:
        self.released = True


def _full_zone() -> Zone:
    return Zone(
        zone_id="machinery_zone",
        name="Machinery Zone",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)),
        required_ppe=("helmet", "vest", "boots"),
        risk_level=1,
    )


def _corner_zone() -> Zone:
    return Zone(
        zone_id="welding_zone",
        name="Welding Zone",
        polygon=(Point(0.0, 0.0), Point(0.1, 0.0), Point(0.1, 0.1), Point(0.0, 0.1)),
        required_ppe=("helmet", "vest", "goggles"),
        risk_level=2,
    )


def _person() -> Detection:
    return Detection("person", 0.9, (20.0, 4.0, 40.0, 44.0))


def _helmet_and_vest() -> list[Detection]:
    return [
        _person(),
        Detection("helmet", 0.9, (24.0, 6.0, 36.0, 14.0)),
        Detection("vest", 0.9, (22.0, 18.0, 38.0, 30.0)),
    ]


def test_inside_zone_escalates_after_the_hold(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    detector = ScriptedDetector([[_person()]])
    pipeline = SmartVisionPipeline(
        settings,
        [_full_zone()],
        detector,
        project_root=tmp_path,
    )
    early = pipeline.process_frame(frame, timestamp=0.0)
    assert early.alerts == []
    assert early.workers[0].status.value == "WARNING"
    assert early.workers[0].missing_ppe == ("helmet", "vest", "boots")
    assert early.workers[0].zone is not None
    assert early.kpis["workers_in_zones"] == 1
    assert early.kpis["active_critical_violations"] == 0
    assert early.kpis["zone_counts"] == {"Machinery Zone": 1}
    assert early.annotated_frame is not None
    assert early.annotated_frame.shape == frame.shape
    assert frame.sum() == 0

    pipeline.process_frame(frame, timestamp=1.0)
    done = pipeline.process_frame(frame, timestamp=2.0)
    assert done.workers[0].track_id == early.workers[0].track_id
    assert done.workers[0].status.value == "CRITICAL"
    assert len(done.alerts) == 1
    assert done.kpis["active_critical_violations"] == 1
    assert Path(done.alerts[0].evidence_path or "").is_file()


def test_partial_box_overlap_puts_the_worker_in_the_zone(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    clipping = Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0))
    pipeline = SmartVisionPipeline(
        settings,
        [_corner_zone()],
        ScriptedDetector([[clipping]]),
        project_root=tmp_path,
    )
    result = pipeline.process_frame(frame, timestamp=0.0)
    assert result.workers[0].zone is not None
    assert result.workers[0].zone.zone_id == "welding_zone"


def test_outside_zone_warns_without_a_timer(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    pipeline = SmartVisionPipeline(
        settings,
        [_corner_zone()],
        ScriptedDetector([[_person()]]),
        project_root=tmp_path,
    )
    result = pipeline.process_frame(frame, timestamp=50.0)
    assert result.workers[0].zone is None
    assert result.workers[0].status.value == "WARNING"
    assert result.alerts == []
    assert result.kpis["workers_in_zones"] == 0
    assert pipeline.violations.count_by_zone() == {}


def test_unselected_ppe_does_not_change_compliance(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    zone = Zone(
        zone_id="helmet_bay",
        name="Helmet Bay",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0)),
        required_ppe=("helmet",),
        risk_level=1,
    )
    detections = _helmet_and_vest() + [Detection("gloves", 0.9, (22.0, 32.0, 30.0, 40.0))]
    pipeline = SmartVisionPipeline(settings, [zone], ScriptedDetector([detections]), project_root=tmp_path)
    result = pipeline.process_frame(frame, timestamp=0.0)
    worker = result.workers[0]
    assert worker.required_ppe == ("helmet",)
    assert worker.worn_ppe == ("helmet",)
    assert worker.missing_ppe == ()
    assert worker.uncertain_ppe == ()
    assert worker.is_compliant
    assert worker.status.value == "OK"
    assert result.alerts == []


def test_compliant_worker_is_ok(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    pipeline = SmartVisionPipeline(
        settings,
        [_corner_zone()],
        ScriptedDetector([_helmet_and_vest()]),
        project_root=tmp_path,
    )
    result = pipeline.process_frame(frame, timestamp=0.0)
    assert result.workers[0].status.value == "OK"
    assert result.workers[0].is_compliant
    assert result.alerts == []


def test_process_image_reports_a_violation_without_waiting(settings: dict, tmp_path: Path) -> None:
    image = tmp_path / "still.png"
    cv2.imwrite(str(image), np.zeros((100, 120, 3), dtype=np.uint8))
    pipeline = SmartVisionPipeline(
        settings,
        [_full_zone()],
        ScriptedDetector([[_person()]]),
        project_root=tmp_path,
    )
    result = pipeline.process_image(image)
    assert result.workers[0].status.value == "CRITICAL"
    assert len(result.alerts) == 1


def test_process_video_skips_frames_and_uses_file_time(settings: dict, tmp_path: Path) -> None:
    detector = ScriptedDetector([[_person()]])
    pipeline = SmartVisionPipeline(
        settings,
        [_full_zone()],
        detector,
        project_root=tmp_path,
    )
    frames = [np.zeros((100, 120, 3), dtype=np.uint8) for _ in range(5)]
    capture = FakeCapture(frames, fps=10.0)
    results = list(pipeline.process_video("clip.mp4", frame_skip=1, capture=capture))
    assert detector.calls == 3
    assert [round(result.timestamp, 2) for result in results] == [0.0, 0.2, 0.4]
    assert capture.released is False


def test_live_sources_and_negative_skip(monkeypatch: pytest.MonkeyPatch, settings: dict, tmp_path: Path) -> None:
    assert is_live_source(0)
    assert is_live_source("rtsp://camera/stream")
    assert not is_live_source("data/samples/site.mp4")
    seen: dict[str, object] = {}

    class _Cap(FakeCapture):
        def __init__(self, source: object) -> None:
            super().__init__([])
            seen["source"] = source

    monkeypatch.setattr("smartvision.pipeline.cv2.VideoCapture", lambda source: _Cap(source))
    monkeypatch.setattr("smartvision.cameras.open_webcam", lambda index: _Cap(index))
    opened = open_video_capture("clip.mp4")
    assert seen["source"] == "clip.mp4"
    opened.release()
    opened = open_video_capture("0")
    assert seen["source"] == 0
    opened.release()
    opened = open_video_capture(1)
    assert seen["source"] == 1
    opened.release()
    pipeline = SmartVisionPipeline(settings, [], ScriptedDetector([[]]), project_root=tmp_path)
    assert list(pipeline.process_video("rtsp://camera/stream", capture=FakeCapture([]))) == []
    with pytest.raises(ValueError, match="frame_skip"):
        list(pipeline.process_video("clip.mp4", frame_skip=-1, capture=FakeCapture([])))
