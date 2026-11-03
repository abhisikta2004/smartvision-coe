"""Photo checks and recorded-video playback.

This session never opens a webcam. The live camera stays on its own
capture. Sample videos and photos use the zones file, which is separate
from the polygon drawn on the webcam.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from smartvision.pipeline import FrameResult, SmartVisionPipeline, Worker
from smartvision.zones import Zone, load_zones

_OpenCapture = Callable[[str], Any]
_PipelineFactory = Callable[[list[Zone]], SmartVisionPipeline]


class ReviewSession:
    """Run still images and video files through the detection pipeline."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        project_root: Path,
        zones_path: Path,
        pipeline_factory: _PipelineFactory | None = None,
        open_capture: _OpenCapture | None = None,
    ) -> None:
        """Store paths and the hooks tests use instead of YOLO.

        Args:
            settings: Parsed ``config/settings.yaml``.
            project_root: Base for models and the alerts database.
            zones_path: Zone file for photos and recorded video.
            pipeline_factory: Builds a pipeline for one zone list. When
                omitted, the real detector is loaded on the first check.
            open_capture: Optional capture opener. Tests pass a scripted
                capture. Files and stream URLs use OpenCV otherwise.
        """
        self._settings = settings
        self._root = project_root
        self._zones_path = zones_path
        self._pipeline_factory = pipeline_factory
        self._open_capture = open_capture
        self._detector: Any = None
        self._infer = threading.Lock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._photo: bytes | None = None
        self._photo_body: dict[str, Any] = {}
        self._running = False
        self._finished = False
        self._error: str | None = None
        self._source: str | None = None
        self._last: dict[str, Any] = {}

    def check_photo(
        self,
        path: Path,
        required_ppe: Sequence[str] | None = None,
        zones: list[Zone] | None = None,
    ) -> dict[str, Any]:
        """Check one image against the zones file or zones drawn on it.

        Args:
            path: Image readable by OpenCV.
            required_ppe: PPE the file zones require for this check.
                ``None`` keeps the list stored with each zone. Ignored
                when ``zones`` is given.
            zones: Polygons drawn on this photo. ``None`` uses the zone file.

        Returns:
            Workers, alerts, and KPIs for that image. The annotated JPEG
            is kept for :meth:`photo_jpeg`.

        Raises:
            FileNotFoundError: The image cannot be read.
            RuntimeError: The model weights are missing.
            ValueError: ``required_ppe`` names a class that is not configured.
        """
        pipeline = self._new_pipeline(self._choose_zones(required_ppe, zones))
        try:
            with self._infer:
                result = pipeline.process_image(path)
        finally:
            pipeline.violations.close()
        body = _public_result(result)
        with self._lock:
            self._photo = encode_jpeg(result.annotated_frame)
            self._photo_body = body
        return body

    def photo_jpeg(self) -> bytes | None:
        """Return the last checked image, or ``None`` before the first check."""
        with self._lock:
            return self._photo

    def start_video(
        self,
        source: str,
        required_ppe: Sequence[str] | None = None,
        zones: list[Zone] | None = None,
    ) -> None:
        """Play a video file or a network stream.

        Args:
            source: File path or ``rtsp`` / ``http`` / ``https`` URL.
            required_ppe: PPE the file zones require for this playback.
                ``None`` keeps the list stored with each zone. Ignored
                when ``zones`` is given.
            zones: Polygons drawn on this video. ``None`` uses the zone file.

        Raises:
            ValueError: ``source`` is a webcam index, or ``required_ppe``
                names a class that is not configured. The live camera tab
                owns the webcam.
        """
        text = str(source).strip()
        if text.isdigit():
            raise ValueError("Use the Live camera tab for a webcam.")
        if not text:
            raise ValueError("Choose a video or enter a stream URL.")
        if "://" in text and not text.startswith(("rtsp://", "http://", "https://")):
            raise ValueError("A stream URL must start with rtsp://, http://, or https://.")
        self.stop_video()
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("The video is still stopping.")
        chosen = self._choose_zones(required_ppe, zones)
        with self._lock:
            self._source = text
            self._error = None
            self._finished = False
            self._running = True
            self._last = {}
            self._jpeg = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(chosen,),
            name="smartvision-video",
            daemon=True,
        )
        self._thread.start()

    def stop_video(self) -> None:
        """Stop playback and release the file. A photo check is left as it is."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        self._thread = None
        with self._lock:
            self._running = False

    def latest_jpeg(self) -> bytes | None:
        """Return the newest annotated video frame."""
        with self._lock:
            return self._jpeg

    def video_status(self) -> dict[str, Any]:
        """Return playback progress, KPIs, workers, and any error."""
        with self._lock:
            body = {
                "running": self._running,
                "finished": self._finished,
                "error": self._error,
                "source": self._source,
            }
            body.update(self._last)
            return body

    def _loop(self, zones: list[Zone]) -> None:
        """Read until the file ends or :meth:`stop_video` is called."""
        pipeline: SmartVisionPipeline | None = None
        capture: Any = None
        try:
            pipeline = self._new_pipeline(zones)
            if self._open_capture is not None:
                capture = self._open_capture(self._source or "")
            iterator = pipeline.process_video(self._source or "", capture=capture)
            try:
                while not self._stop.is_set():
                    with self._infer:
                        try:
                            result = next(iterator)
                        except StopIteration:
                            break
                    self._remember(result)
            finally:
                iterator.close()
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            if pipeline is not None:
                pipeline.violations.close()
            with self._lock:
                self._running = False
                self._finished = True

    def _remember(self, result: FrameResult) -> None:
        """Keep the overlay and the public summary of one video frame."""
        encoded = encode_jpeg(result.annotated_frame)
        body = _public_result(result)
        fresh = [_alert_row(alert) for alert in result.alerts]
        with self._lock:
            previous = list(self._last.get("alerts") or [])
            if encoded:
                self._jpeg = encoded
            self._last = body
            self._last["alerts"] = (fresh + previous)[:8]

    def _load_zones(self) -> list[Zone]:
        """Load the sample-video zone file. A missing file means no zones."""
        if not self._zones_path.is_file():
            return []
        return load_zones(self._zones_path)

    def _choose_zones(self, required_ppe: Sequence[str] | None, zones: list[Zone] | None) -> list[Zone]:
        """Pick drawn zones, or the zone file when nothing was drawn.

        Args:
            required_ppe: PPE applied to the file zones.
            zones: Polygons drawn on this photo or video.

        Returns:
            The zones for this check.
        """
        if zones is not None:
            return list(zones)
        return self._zones_for(required_ppe)

    def _zones_for(self, required_ppe: Sequence[str] | None) -> list[Zone]:
        """Return the zone file, optionally with one required-PPE list.

        Args:
            required_ppe: Classes checked on the photo or video tab.
                ``None`` leaves each zone's saved list unchanged.

        Returns:
            Zones for this check. Polygons stay as they are on disk.

        Raises:
            ValueError: A name is not one of the configured PPE classes.
        """
        zones = self._load_zones()
        if required_ppe is None:
            return zones
        allowed = {str(name) for name in self._settings["classes"]["ppe"]}
        required = tuple(str(name) for name in required_ppe)
        unknown = [name for name in required if name not in allowed]
        if unknown:
            raise ValueError(f"Unknown PPE: {', '.join(unknown)}")
        return [replace(zone, required_ppe=required) for zone in zones]

    def _new_pipeline(self, zones: list[Zone]) -> SmartVisionPipeline:
        """Build a pipeline, loading YOLO once when no factory was injected."""
        if self._pipeline_factory is not None:
            return self._pipeline_factory(zones)
        if self._detector is None:
            from smartvision.detector import PPEDetector

            self._detector = PPEDetector(self._settings, project_root=self._root)
        return SmartVisionPipeline(
            self._settings,
            zones,
            self._detector,
            project_root=self._root,
            show_zones=False,
        )


def _public_result(result: FrameResult) -> dict[str, Any]:
    """JSON-ready workers, alerts, and KPIs for one frame."""
    return {
        "kpis": {
            "workers_in_zones": int(result.kpis.get("workers_in_zones", 0)),
            "active_critical_violations": int(result.kpis.get("active_critical_violations", 0)),
            "zone_counts": dict(result.kpis.get("zone_counts") or {}),
        },
        "workers": [_worker_row(worker) for worker in result.workers],
        "alerts": [_alert_row(alert) for alert in result.alerts],
    }


def _worker_row(worker: Worker) -> dict[str, Any]:
    """One table row for a tracked person."""
    detected = list(dict.fromkeys([*worker.worn_ppe, *worker.seen_ppe]))
    return {
        "track_id": worker.track_id,
        "zone": worker.zone.name if worker.zone is not None else "Outside",
        "required_ppe": list(worker.required_ppe),
        "detected_ppe": detected,
        "worn_ppe": list(worker.worn_ppe),
        "missing_ppe": [name for name in worker.missing_ppe if name not in set(detected)],
        "uncertain_ppe": list(worker.uncertain_ppe),
        "status": worker.status.value,
    }


def _alert_row(alert: Any) -> dict[str, Any]:
    """One alert row without a filesystem path."""
    evidence = Path(alert.evidence_path).name if alert.evidence_path else ""
    return {
        "timestamp": float(alert.timestamp),
        "track_id": alert.track_id,
        "zone": alert.zone_name or alert.zone_id or "",
        "missing_ppe": list(alert.missing_ppe),
        "evidence": evidence,
    }


def encode_jpeg(frame: np.ndarray | None) -> bytes:
    """Encode a BGR frame, shrinking wide images so the page stays light."""
    if frame is None:
        return b""
    height, width = frame.shape[:2]
    shown = frame
    if width > 960:
        scale = 960 / width
        shown = cv2.resize(frame, (960, max(1, int(round(height * scale)))))
    ok, encoded = cv2.imencode(".jpg", shown, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return b""
    return encoded.tobytes()
