"""SmartVision dashboard.

Run from the repository root:

    .venv/bin/python -m uvicorn app.live_server:app --host 127.0.0.1 --port 8000

The live camera stays open while a zone is drawn and while detection runs.
Photo checks, recorded video, alert history, and the sample-video zone file
are on the same page. Those use ``config/zones.yaml``. The webcam polygon
stays in ``outputs/webcam_zone.yaml``.
"""

from __future__ import annotations

import csv
import io
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.zone_helper import format_zones_yaml, unique_id  # noqa: E402
from smartvision.alerts import ViolationManager  # noqa: E402
from smartvision.library import (  # noqa: E402
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    list_media,
    media_zones_path,
    read_media_zones,
    reference_frame,
    resolve_media,
    store_upload,
    write_media_zones,
)
from smartvision.live_monitor import LiveMonitor  # noqa: E402
from smartvision.pipeline import load_settings  # noqa: E402
from smartvision.review import ReviewSession, encode_jpeg  # noqa: E402
from smartvision.zones import Point, Zone, load_zones  # noqa: E402

PAGE = ROOT / "app" / "live.html"


class CameraIn(BaseModel):
    """Which webcam to open."""

    index: int = Field(ge=0, le=8)


class ZoneIn(BaseModel):
    """A polygon drawn on the live picture."""

    name: str
    points: list[tuple[float, float]]
    ppe: list[str]

    @field_validator("points", mode="before")
    @classmethod
    def point_pairs(cls, value: object) -> object:
        """Accept ``[x, y]`` pairs and ``{x, y}`` corners from the page."""
        if not isinstance(value, list):
            return value
        pairs: list[object] = []
        for item in value:
            if isinstance(item, dict):
                pairs.append((item.get("x"), item.get("y")))
            else:
                pairs.append(item)
        return pairs


class MediaIn(BaseModel):
    """A sample or uploaded file chosen in the dashboard."""

    kind: str = "sample"
    name: str = ""
    url: str = ""
    frame: int = 0
    ppe: list[str] | None = None
    zones: list[StoredZone] | None = None


class MediaZonesIn(BaseModel):
    """Zones drawn on one sample or upload."""

    kind: str
    name: str
    zones: list["StoredZone"]


class StoredZone(BaseModel):
    """One zone written to the sample-video zone file."""

    name: str
    points: list[tuple[float, float]]
    ppe: list[str] = Field(default_factory=list)
    risk: int = Field(default=1, ge=0, le=100)

    @field_validator("points", mode="before")
    @classmethod
    def point_pairs(cls, value: object) -> object:
        """Accept ``[x, y]`` pairs and ``{x, y}`` corners."""
        return _pair_points(value)


class ZonesFileIn(BaseModel):
    """The full zone list for photos and recorded video."""

    zones: list[StoredZone]


def create_app(
    monitor: LiveMonitor | None = None,
    review: ReviewSession | None = None,
    *,
    settings: dict[str, Any] | None = None,
    project_root: Path | None = None,
    samples_dir: Path | None = None,
    uploads_dir: Path | None = None,
    zones_file: Path | None = None,
    eval_csv: Path | None = None,
) -> FastAPI:
    """Build the dashboard.

    Args:
        monitor: Live camera session. The default opens the real webcam.
        review: Photo and video session. The default loads YOLO on first use.
        settings: Parsed settings. The repository file is used when omitted.
        project_root: Base for models, alerts, and evidence.
        samples_dir: Folder of sample images and videos.
        uploads_dir: Folder that receives uploads.
        zones_file: Zone file for photos and recorded video.
        eval_csv: Optional metrics table for the model page.

    Returns:
        The FastAPI application.
    """
    root = project_root or ROOT
    parsed = settings if settings is not None else load_settings(ROOT / "config" / "settings.yaml")
    samples = samples_dir or (root / "data" / "samples")
    uploads = uploads_dir or (root / "outputs" / "uploads")
    drawn_zones_dir = root / "outputs" / "media_zones"
    zone_path = zones_file or (root / "config" / "zones.yaml")
    metrics = eval_csv or (root / "outputs" / "eval_results.csv")
    session = monitor if monitor is not None else LiveMonitor(parsed, project_root=root)
    clips = review if review is not None else ReviewSession(parsed, project_root=root, zones_path=zone_path)
    app = FastAPI(title="SmartVision")
    app.state.monitor = session
    app.state.review = clips

    @app.get("/")
    def page() -> FileResponse:
        """Serve the dashboard. A refresh always picks up the latest page."""
        return FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/api/config")
    def config() -> dict[str, object]:
        """PPE choices and camera labels for the page."""
        classes = parsed["classes"]
        return {
            "ppe": list(classes["ppe"]),
            "display_names": dict(classes.get("display_names", {})),
            "default_required_ppe": list(parsed["compliance"]["default_required_ppe"]),
            "status_seconds": float(parsed["alerts"].get("live_status_seconds", 5.0)),
            "cameras": [
                {"index": 0, "label": "iPhone camera"},
                {"index": 1, "label": "Laptop webcam"},
            ],
        }

    @app.post("/api/camera")
    def open_camera(body: CameraIn) -> dict[str, object]:
        """Open a webcam. A later zone save keeps this capture."""
        try:
            session.start(str(body.index))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return session.snapshot()

    @app.post("/api/camera/stop")
    def close_camera() -> dict[str, object]:
        """Release the webcam and clear the zone."""
        session.stop()
        return session.snapshot()

    @app.get("/api/stream")
    def stream() -> StreamingResponse:
        """Stream the latest JPEG without restarting the camera."""
        return StreamingResponse(session_frames(session), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/zone")
    def save_zone(body: ZoneIn) -> dict[str, object]:
        """Save the polygon and start detection on the open camera."""
        try:
            session.save_zone(body.name, body.points, body.ppe)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return session.snapshot()

    @app.post("/api/zone/edit")
    def edit_zone() -> dict[str, object]:
        """Stop detection and allow a new polygon. The camera stays open."""
        session.edit_zone()
        return session.snapshot()

    @app.post("/api/zone/reset")
    def reset_zone() -> dict[str, object]:
        """Clear the polygon. The camera stays open."""
        session.reset_zone()
        return session.snapshot()

    @app.get("/api/status")
    def status() -> dict[str, object]:
        """Return the latest safety panel."""
        return session.snapshot()

    @app.get("/api/library")
    def library() -> dict[str, list[str]]:
        """List sample and uploaded images and videos."""
        return {
            "videos": sorted(set(list_media(samples, VIDEO_SUFFIXES) + list_media(uploads, VIDEO_SUFFIXES))),
            "images": sorted(set(list_media(samples, IMAGE_SUFFIXES) + list_media(uploads, IMAGE_SUFFIXES))),
            "sample_videos": list_media(samples, VIDEO_SUFFIXES),
            "sample_images": list_media(samples, IMAGE_SUFFIXES),
            "upload_videos": list_media(uploads, VIDEO_SUFFIXES),
            "upload_images": list_media(uploads, IMAGE_SUFFIXES),
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile) -> dict[str, str]:
        """Store an image or video. This does not start detection."""
        payload = await file.read()
        if len(payload) > 100_000_000:
            raise HTTPException(status_code=413, detail="The file is larger than 100 MB.")
        try:
            stored = store_upload(uploads, file.filename or "", payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        suffix = Path(stored).suffix.lower()
        return {"name": stored, "kind": "image" if suffix in IMAGE_SUFFIXES else "video"}

    @app.get("/api/media")
    def media_file(kind: str, name: str) -> FileResponse:
        """Serve one uploaded or sample file so the page can play it."""
        try:
            path = _media_path(
                MediaIn(kind=kind, name=name),
                samples,
                uploads,
                IMAGE_SUFFIXES | VIDEO_SUFFIXES,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(path)

    @app.post("/api/photo")
    def check_photo(body: MediaIn) -> dict[str, Any]:
        """Check one sample or uploaded image against the automatic work area."""
        try:
            path = _media_path(body, samples, uploads, IMAGE_SUFFIXES)
            return clips.check_photo(path, None, _auto_zone(parsed, body.ppe))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/photo.jpg")
    def photo_jpeg() -> Response:
        """Return the last checked image."""
        jpeg = clips.photo_jpeg()
        if not jpeg:
            raise HTTPException(status_code=404, detail="Check an image first.")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.post("/api/video")
    def play_video(body: MediaIn) -> dict[str, Any]:
        """Play a sample, an upload, or a network stream inside the automatic work area."""
        try:
            zones = _auto_zone(parsed, body.ppe)
            if body.kind == "url":
                clips.start_video(body.url, None, zones)
            else:
                path = _media_path(body, samples, uploads, VIDEO_SUFFIXES)
                clips.start_video(str(path), None, zones)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return clips.video_status()

    @app.post("/api/video/stop")
    def stop_video() -> dict[str, Any]:
        """Stop recorded-video playback. The webcam is left alone."""
        clips.stop_video()
        return clips.video_status()

    @app.get("/api/video/status")
    def video_status() -> dict[str, Any]:
        """Return the latest video KPIs, workers, and alerts."""
        return clips.video_status()

    @app.get("/api/video/stream")
    def video_stream() -> StreamingResponse:
        """Stream annotated frames from the recorded video."""
        return StreamingResponse(
            jpeg_frames(clips.latest_jpeg),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/media-zones")
    def media_zones(kind: str, name: str) -> dict[str, Any]:
        """Return zones previously drawn on one sample or upload."""
        try:
            saved = read_media_zones(media_zones_path(drawn_zones_dir, kind, name))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"zones": saved}

    @app.post("/api/media-zones")
    def save_media_zones(body: MediaZonesIn) -> dict[str, Any]:
        """Remember zones drawn on one sample or upload."""
        try:
            zones = _zones_from_request(body.zones, parsed)
            _store_drawn_zones(drawn_zones_dir, body.kind, body.name, zones)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"zones": [_zone_json(zone) for zone in zones]}

    @app.get("/api/reference.jpg")
    def reference(kind: str = "sample", name: str = "", frame: int = 0) -> Response:
        """Return one frame so a sample-video zone can be drawn on it."""
        try:
            path = _media_path(MediaIn(kind=kind, name=name), samples, uploads, IMAGE_SUFFIXES | VIDEO_SUFFIXES)
            jpeg = encode_jpeg(reference_frame(path, frame))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not jpeg:
            raise HTTPException(status_code=400, detail="Could not encode that frame.")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/zones")
    def read_zones() -> dict[str, Any]:
        """Return the zones used for photos and recorded video."""
        zones = load_zones(zone_path) if zone_path.is_file() else []
        return {"zones": [_zone_json(zone) for zone in zones]}

    @app.post("/api/zones")
    def write_zones(body: ZonesFileIn) -> dict[str, Any]:
        """Replace the sample-video zone file. The webcam zone is not changed."""
        try:
            zones = _zones_from_request(body.zones, parsed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if zone_path.is_file():
            shutil.copy2(zone_path, zone_path.with_suffix(zone_path.suffix + ".bak"))
        zone_path.parent.mkdir(parents=True, exist_ok=True)
        zone_path.write_text(format_zones_yaml(zones), encoding="utf-8")
        return {"zones": [_zone_json(zone) for zone in zones]}

    @app.get("/api/alerts")
    def alerts(zone: str = "") -> dict[str, Any]:
        """Return stored critical alerts, newest first."""
        rows = _query_alerts(parsed, root, zone or None)
        return {"alerts": rows}

    @app.get("/api/alerts.csv")
    def alerts_csv(zone: str = "") -> Response:
        """Download the alert table as CSV."""
        rows = _query_alerts(parsed, root, zone or None)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Timestamp", "Worker", "Zone", "Missing PPE", "Evidence"])
        for row in rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["track_id"],
                    row["zone"],
                    _ppe_text(row["missing_ppe"], parsed),
                    row["evidence"],
                ]
            )
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=alerts.csv"},
        )

    @app.get("/api/evidence/{name}")
    def evidence(name: str) -> FileResponse:
        """Serve one evidence JPEG from the alerts folder."""
        folder = _evidence_dir(parsed, root)
        try:
            path = resolve_media(folder, name, {".jpg", ".jpeg", ".png"})
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Evidence file not found.") from exc
        return FileResponse(path)

    @app.get("/api/about")
    def about() -> dict[str, Any]:
        """Describe the models and return the saved test-split metrics."""
        rows: list[dict[str, str]] = []
        if metrics.is_file():
            with metrics.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        return {
            "summary": (
                "People come from pretrained COCO YOLOv8s. Helmet, vest, gloves, "
                "boots, and goggles come from models/best.pt."
            ),
            "metrics": rows,
        }

    return app


def session_frames(monitor: LiveMonitor):
    """Yield multipart JPEG chunks from the latest camera frame."""
    yield from jpeg_frames(monitor.latest_jpeg)


def jpeg_frames(latest: Callable[[], bytes | None]):
    """Yield multipart JPEG chunks from a latest-frame callable."""
    while True:
        jpeg = latest()
        if not jpeg:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.05)


def _pair_points(value: object) -> object:
    """Normalize polygon corners to pairs before validation."""
    if not isinstance(value, list):
        return value
    pairs: list[object] = []
    for item in value:
        if isinstance(item, dict):
            pairs.append((item.get("x"), item.get("y")))
        else:
            pairs.append(item)
    return pairs


def _auto_zone(settings: dict[str, Any], ppe: list[str] | None) -> list[Zone]:
    """Build the zone that covers a photo or recorded video.

    Args:
        settings: Parsed settings. The polygon and name come from
            ``review.auto_zone``.
        ppe: Checklist chosen on the page. ``None`` uses
            ``compliance.default_required_ppe``.

    Returns:
        One zone covering the frame configured in settings.
    """
    spec = settings["review"]["auto_zone"]
    required = (
        list(settings["compliance"]["default_required_ppe"]) if ppe is None else list(ppe)
    )
    item = StoredZone(
        name=str(spec["name"]),
        points=[(float(point[0]), float(point[1])) for point in spec["points"]],
        ppe=required,
        risk=int(spec.get("risk", 1)),
    )
    return _zones_from_request([item], settings)


def _store_drawn_zones(directory: Path, kind: str, name: str, zones: list[Zone]) -> None:
    """Write the zones drawn on one sample or upload."""
    if kind not in {"sample", "upload"}:
        return
    write_media_zones(media_zones_path(directory, kind, name), [_zone_json(zone) for zone in zones])


def _media_path(body: MediaIn, samples: Path, uploads: Path, suffixes: set[str]) -> Path:
    """Resolve a sample or upload name inside its library folder."""
    if body.kind not in {"sample", "upload"}:
        raise ValueError("Choose a sample or an upload.")
    folder = samples if body.kind == "sample" else uploads
    return resolve_media(folder, body.name, suffixes)


def _zones_from_request(items: list[StoredZone], settings: dict[str, Any]) -> list[Zone]:
    """Validate and build zones for the sample-video file."""
    allowed = {str(name) for name in settings["classes"]["ppe"]}
    taken: set[str] = set()
    zones: list[Zone] = []
    for item in items:
        name = item.name.strip()
        if not name:
            raise ValueError("Enter a zone name.")
        if len(item.points) < 3:
            raise ValueError("A zone needs at least 3 points.")
        polygon: list[Point] = []
        for x_pos, y_pos in item.points:
            if not 0.0 <= float(x_pos) <= 1.0 or not 0.0 <= float(y_pos) <= 1.0:
                raise ValueError("Zone points must stay between 0 and 1.")
            polygon.append(Point(float(x_pos), float(y_pos)))
        unknown = [ppe for ppe in item.ppe if ppe not in allowed]
        if unknown:
            raise ValueError(f"Unknown PPE: {', '.join(unknown)}")
        zone_id = unique_id(name, taken)
        taken.add(zone_id)
        zones.append(
            Zone(
                zone_id=zone_id,
                name=name,
                polygon=tuple(polygon),
                required_ppe=tuple(item.ppe),
                risk_level=int(item.risk),
            )
        )
    return zones


def _zone_json(zone: Zone) -> dict[str, Any]:
    """JSON for one stored zone."""
    return {
        "id": zone.zone_id,
        "name": zone.name,
        "risk": zone.risk_level,
        "ppe": list(zone.required_ppe),
        "points": [{"x": point.x, "y": point.y} for point in zone.polygon],
    }


def _ppe_text(names: list[str], settings: dict[str, Any]) -> str:
    """Join PPE class names with their display labels.

    Args:
        names: Canonical class names.
        settings: Parsed settings. ``classes.display_names`` supplies labels.

    Returns:
        A comma-separated label list, or ``None`` when ``names`` is empty.
    """
    labels = {str(key): str(value) for key, value in settings["classes"].get("display_names", {}).items()}
    shown = [labels.get(str(name), str(name)) for name in names]
    return ", ".join(shown) if shown else "None"


def _query_alerts(settings: dict[str, Any], root: Path, zone: str | None) -> list[dict[str, Any]]:
    """Read alerts and close the short-lived database connection."""
    manager = ViolationManager(settings, project_root=root)
    try:
        rows = manager.query_alerts(zone=zone, limit=500)
    finally:
        manager.close()
    return [
        {
            "timestamp": alert.timestamp,
            "track_id": alert.track_id,
            "zone": alert.zone_name or alert.zone_id or "",
            "missing_ppe": list(alert.missing_ppe),
            "evidence": Path(alert.evidence_path).name if alert.evidence_path else "",
        }
        for alert in rows
    ]


def _evidence_dir(settings: dict[str, Any], root: Path) -> Path:
    """Return the evidence folder, relative paths starting at ``root``."""
    folder = Path(str(settings["alerts"]["evidence_dir"]))
    if not folder.is_absolute():
        folder = root / folder
    return folder


app = create_app()
