"""Photo, video, and library checks. No camera and no model weights."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.live_server import create_app
from smartvision.detector import Detection
from smartvision.library import IMAGE_SUFFIXES, resolve_media, store_upload
from smartvision.live_monitor import LiveMonitor
from smartvision.pipeline import SmartVisionPipeline, Worker, WorkerStatus
from smartvision.review import ReviewSession, _worker_row
from smartvision.zones import Zone


class OnceCapture:
    """Returns one frame, then ends."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.reads = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.released or self.reads:
            return False, None
        self.reads += 1
        return True, self.frame.copy()

    def get(self, _prop: int) -> float:
        return 25.0

    def release(self) -> None:
        self.released = True


class ScriptedDetector:
    """Returns one fixed person."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return [Detection("person", 0.9, (0.0, 0.0, 40.0, 40.0))]


def _zones(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "zones:",
                "  - id: machine",
                "    name: Machine 1",
                "    risk_level: 1",
                "    required_ppe: [helmet, vest]",
                "    polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _review(tmp_path: Path, settings: dict, frame: np.ndarray) -> tuple[ReviewSession, list[str]]:
    opened: list[str] = []

    def factory(zones: list[Zone]) -> SmartVisionPipeline:
        return SmartVisionPipeline(settings, zones, ScriptedDetector(), project_root=tmp_path)

    def opener(source: str) -> OnceCapture:
        opened.append(source)
        return OnceCapture(frame)

    review = ReviewSession(
        settings,
        project_root=tmp_path,
        zones_path=tmp_path / "zones.yaml",
        pipeline_factory=factory,
        open_capture=opener,
    )
    return review, opened


def _client(tmp_path: Path, settings: dict, frame: np.ndarray) -> TestClient:
    _zones(tmp_path / "zones.yaml")
    samples = tmp_path / "samples"
    samples.mkdir()
    cv2.imwrite(str(samples / "person.png"), frame)
    review, _opened = _review(tmp_path, settings, frame)

    def factory(zones: list[Zone]) -> SmartVisionPipeline:
        return SmartVisionPipeline(settings, zones, ScriptedDetector(), project_root=tmp_path)

    monitor = LiveMonitor(
        settings,
        project_root=tmp_path,
        zones_path=tmp_path / "webcam_zone.yaml",
        open_capture=lambda _source: OnceCapture(frame),
        pipeline_factory=factory,
    )
    return TestClient(
        create_app(
            monitor,
            review,
            settings=settings,
            project_root=tmp_path,
            samples_dir=samples,
            uploads_dir=tmp_path / "uploads",
            zones_file=tmp_path / "zones.yaml",
            eval_csv=tmp_path / "eval.csv",
        )
    )


def test_saved_zones_stay_with_one_upload(tmp_path: Path) -> None:
    from smartvision.library import media_zones_path, read_media_zones, write_media_zones

    path = media_zones_path(tmp_path, "upload", "booth.jpg")
    write_media_zones(path, [{"name": "Booth", "ppe": ["helmet"], "points": [{"x": 0.1, "y": 0.2}]}])
    assert read_media_zones(path)[0]["name"] == "Booth"
    with pytest.raises(ValueError):
        media_zones_path(tmp_path, "other", "booth.jpg")


def test_upload_names_stay_inside_the_upload_folder(tmp_path: Path) -> None:
    folder = tmp_path / "uploads"
    stored = store_upload(folder, "../secret shot.jpg", b"\xff\xd8\xff")
    assert stored == "secret shot.jpg".replace(" ", "_")
    assert (folder / stored).is_file()
    assert resolve_media(folder, stored, IMAGE_SUFFIXES) == (folder / stored).resolve()
    with pytest.raises(ValueError):
        store_upload(folder, "notes.txt", b"hello")
    with pytest.raises(ValueError):
        resolve_media(folder, "..", IMAGE_SUFFIXES)


def test_uploaded_video_is_served_for_playback(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    client = _client(tmp_path, settings, frame)
    stored = client.post("/api/upload", files={"file": ("clip.mp4", b"video-bytes", "video/mp4")})
    assert stored.status_code == 200
    played = client.get("/api/media", params={"kind": "upload", "name": stored.json()["name"]})
    assert played.status_code == 200
    assert played.content == b"video-bytes"
    missing = client.get("/api/media", params={"kind": "upload", "name": "../secret.mp4"})
    assert missing.status_code in {400, 404}


def test_video_playback_does_not_open_a_webcam(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    _zones(tmp_path / "zones.yaml")
    review, opened = _review(tmp_path, settings, frame)
    with pytest.raises(ValueError, match="Live camera"):
        review.start_video("0")
    assert opened == []
    review.start_video(str(tmp_path / "clip.mp4"))
    deadline = time.monotonic() + 2
    status = review.video_status()
    while time.monotonic() < deadline and not status["finished"]:
        time.sleep(0.05)
        status = review.video_status()
    assert opened == [str(tmp_path / "clip.mp4")]
    assert status["finished"] is True
    assert status["workers"]
    assert status["workers"][0]["zone"] == "Machine 1"
    assert "helmet" in status["workers"][0]["missing_ppe"]
    review.stop_video()
    review.start_video(str(tmp_path / "clip.mp4"), required_ppe=["boots"])
    deadline = time.monotonic() + 2
    status = review.video_status()
    while time.monotonic() < deadline and not status["finished"]:
        time.sleep(0.05)
        status = review.video_status()
    assert status["workers"][0]["missing_ppe"] == ["boots"]
    review.stop_video()


def test_config_uses_the_repository_settings(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    def factory(zones: list[Zone]) -> SmartVisionPipeline:
        return SmartVisionPipeline(settings, zones, ScriptedDetector(), project_root=tmp_path)

    monitor = LiveMonitor(
        settings,
        project_root=tmp_path,
        zones_path=tmp_path / "webcam_zone.yaml",
        open_capture=lambda _source: OnceCapture(frame),
        pipeline_factory=factory,
    )
    body = TestClient(create_app(monitor)).get("/api/config").json()
    assert "helmet" in body["ppe"]
    assert body["cameras"][0]["label"] == "iPhone camera"


def test_dashboard_photo_video_alerts_and_zones(tmp_path: Path, settings: dict, frame: np.ndarray) -> None:
    client = _client(tmp_path, settings, frame)
    page = client.get("/")
    assert page.status_code == 200
    assert "helmet" in client.get("/api/config").json()["ppe"]
    assert "Images" in page.text
    assert "Check image" in page.text
    assert "Alerts history" in page.text
    assert "Draw polygon zone" in page.text

    library = client.get("/api/library")
    assert "person.png" in library.json()["sample_images"]

    checked = client.post("/api/photo", json={"kind": "sample", "name": "person.png"})
    assert checked.status_code == 200
    assert checked.json()["alerts"]
    assert checked.json()["workers"][0]["missing_ppe"]
    chosen = client.post(
        "/api/photo",
        json={"kind": "sample", "name": "person.png", "ppe": ["goggles"]},
    )
    assert chosen.status_code == 200
    assert chosen.json()["workers"][0]["missing_ppe"] == ["goggles"]
    drawn = client.post(
        "/api/photo",
        json={
            "kind": "sample",
            "name": "person.png",
            "ppe": ["boots"],
            "zones": [
                {"name": "Booth", "ppe": ["helmet"], "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
            ],
        },
    )
    assert drawn.status_code == 200
    assert drawn.json()["workers"][0]["zone"] == "Work area"
    assert drawn.json()["workers"][0]["missing_ppe"] == ["boots"]
    remembered = client.get("/api/media-zones", params={"kind": "sample", "name": "person.png"})
    assert remembered.json()["zones"] == []
    rejected = client.post(
        "/api/photo",
        json={"kind": "sample", "name": "person.png", "ppe": ["hardhat"]},
    )
    assert rejected.status_code == 400
    picture = client.get("/api/photo.jpg")
    assert picture.status_code == 200
    assert picture.content.startswith(b"\xff\xd8")

    escaped = client.post("/api/photo", json={"kind": "sample", "name": "../zones.yaml"})
    assert escaped.status_code in {400, 404}

    history = client.get("/api/alerts")
    assert history.status_code == 200
    assert history.json()["alerts"]
    evidence = history.json()["alerts"][0]["evidence"]
    assert evidence
    assert client.get(f"/api/evidence/{evidence}").status_code == 200
    assert "Worker" in client.get("/api/alerts.csv").text

    saved = client.post(
        "/api/zones",
        json={
            "zones": [
                {
                    "name": "Loading bay",
                    "risk": 2,
                    "ppe": ["helmet", "vest"],
                    "points": [[0.0, 0.0], [1.0, 0.0], [0.4, 1.0]],
                }
            ]
        },
    )
    assert saved.status_code == 200
    assert saved.json()["zones"][0]["name"] == "Loading bay"
    text = (tmp_path / "zones.yaml").read_text(encoding="utf-8")
    assert "Loading bay" in text
    assert (tmp_path / "zones.yaml.bak").is_file()

    metrics = tmp_path / "eval.csv"
    metrics.write_text("class,precision\nhelmet,0.900\n", encoding="utf-8")
    about = client.get("/api/about")
    assert about.json()["metrics"][0]["class"] == "helmet"


def test_detected_ppe_includes_a_vest_box_on_the_worker() -> None:
    """The table lists a vest drawn on the person even when the band rejected it."""
    worker = Worker(
        track_id=1,
        bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
        zone=None,
        missing_ppe=("vest",),
        worn_ppe=(),
        status=WorkerStatus.CRITICAL,
        is_compliant=False,
        required_ppe=("helmet", "vest"),
        seen_ppe=("vest",),
    )
    row = _worker_row(worker)
    assert row["detected_ppe"] == ["vest"]
    assert row["missing_ppe"] == []
