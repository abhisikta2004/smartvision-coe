"""Tests for the video runner. The detector is not loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smartvision.alerts import Alert
from smartvision.pipeline import FrameResult, Worker, WorkerStatus
from scripts.run_video import output_path, parse_args, run


class _Capture:
    def __init__(self, source: object) -> None:
        self.source = source

    def get(self, _prop: int) -> float:
        return 25.0

    def release(self) -> None:
        return None


class _Pipeline:
    def process_video(self, source: str, frame_skip: int | None = None, capture: object | None = None):
        del source, capture
        assert frame_skip == 2
        worker = Worker(
            track_id=3,
            bbox_xyxy=(1, 1, 8, 8),
            zone=None,
            missing_ppe=("helmet",),
            worn_ppe=(),
            status=WorkerStatus.WARNING,
            is_compliant=False,
        )
        alert = Alert(
            track_id=3,
            zone_id="welding_zone",
            missing_ppe=("goggles",),
            evidence_path="/tmp/evidence.jpg",
            timestamp=1.0,
            zone_name="Welding Zone",
        )
        yield FrameResult(
            workers=[worker],
            alerts=[alert],
            annotated_frame=np.zeros((32, 32, 3), dtype=np.uint8),
            timestamp=1.0,
        )


def test_output_names_and_skip_flag(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("scripts.run_video.open_video_capture", lambda source: _Capture(source))
    args = parse_args(["data/samples/site.mp4", "--skip", "2"])
    assert args.skip == 2
    destination = tmp_path / "site_annotated.mp4"
    run("data/samples/site.mp4", _Pipeline(), destination, args.skip)
    printed = capsys.readouterr().out
    assert "WARNING worker=3" in printed
    assert "CRITICAL worker=3 zone=Welding Zone missing=goggles" in printed
    assert destination.is_file()
    assert output_path("0", tmp_path).name == "webcam_annotated.mp4"
    assert output_path("rtsp://camera/stream", tmp_path).name == "annotated.mp4"
