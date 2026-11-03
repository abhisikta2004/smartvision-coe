"""Webcam session for the live page.

One capture stays open while the operator draws a zone and while detection
runs. Saving a zone swaps the polygon and turns monitoring on. It does not
release the camera. Frames are numpy images, so tests can run without a
device.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from smartvision.live_status import StatusBoard, refresh_board
from smartvision.pipeline import SmartVisionPipeline, load_settings
from smartvision.zones import Point, Zone, draw_zones, save_zones

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LiveMonitor:
    """Hold the open camera, the saved zone, and the latest JPEG."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        project_root: Path | None = None,
        zones_path: Path | None = None,
        open_capture: Callable[[str], Any] | None = None,
        pipeline_factory: Callable[[list[Zone]], SmartVisionPipeline] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Store settings and the hooks tests use instead of a real camera.

        Args:
            settings: Parsed ``config/settings.yaml``. Loaded from disk
                when omitted.
            project_root: Base for the zone file and model paths.
            zones_path: Where the webcam polygon is written.
            open_capture: ``(source) -> capture``. Defaults to the project
                webcam opener.
            pipeline_factory: Builds the detection pipeline when a zone is
                saved. Tests pass a scripted detector here.
            clock: Monotonic clock for the safety-panel interval.
        """
        self._root = project_root or PROJECT_ROOT
        self._settings = settings if settings is not None else load_settings()
        self._zones_path = zones_path or (self._root / "outputs" / "webcam_zone.yaml")
        self._open_capture = open_capture
        self._pipeline_factory = pipeline_factory
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._infer = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any = None
        self._source: str | None = None
        self._pipeline: SmartVisionPipeline | None = None
        self._zones: list[Zone] = []
        self._monitoring = False
        self._board = StatusBoard()
        self._t0 = self._clock()
        self._jpeg: bytes | None = None
        self._error: str | None = None

    def start(self, source: str) -> None:
        """Open a webcam and keep reading it.

        Args:
            source: User camera index. ``0`` is the iPhone and ``1`` is
                the laptop.

        Raises:
            RuntimeError: The camera cannot be opened.
        """
        text = str(source)
        with self._lock:
            same = self._thread is not None and self._thread.is_alive() and self._source == text
        if same:
            return
        self._stop_thread()
        capture = self._open(text)
        if not capture.isOpened():
            message = getattr(capture, "open_error", None) or f"Could not open camera {text}"
            capture.release()
            raise RuntimeError(str(message))
        with self._lock:
            self._capture = capture
            self._source = text
            self._error = None
            self._t0 = self._clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="smartvision-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Release the camera and drop the zone."""
        self._stop_thread()
        with self._infer:
            pipeline = self._pipeline
            self._pipeline = None
            self._zones = []
            self._monitoring = False
            self._board = StatusBoard()
            self._jpeg = None
        if pipeline is not None:
            pipeline.violations.close()
        if self._zones_path.exists():
            self._zones_path.unlink()

    def save_zone(self, name: str, points: list[tuple[float, float]], ppe: list[str]) -> Zone:
        """Store one polygon and start detection on the open camera.

        Args:
            name: Zone label, such as ``Machine 1``.
            points: Normalized corners, each ``(x, y)`` in ``[0, 1]``.
            ppe: Required class names from settings.

        Returns:
            The zone now used for monitoring.

        Raises:
            RuntimeError: The camera is not open.
            ValueError: The name, polygon, or PPE list is not usable.
        """
        zone = _zone_from_request(name, points, ppe, self._settings)
        with self._lock:
            capture = self._capture
        if capture is None or not capture.isOpened():
            raise RuntimeError("The camera is not open.")
        with self._infer:
            if self._pipeline is None:
                self._pipeline = self._build_pipeline([zone])
                self._t0 = self._clock()
            else:
                self._pipeline.set_zones([zone])
                self._pipeline.violations.set_single_open_alert(True)
            self._zones = [zone]
            self._monitoring = True
            self._board = StatusBoard()
            self._error = None
        save_zones(self._zones_path, [zone])
        return zone

    def edit_zone(self) -> None:
        """Return to drawing. The camera and the current polygon stay."""
        with self._infer:
            self._monitoring = False
            self._board = StatusBoard()

    def reset_zone(self) -> None:
        """Remove the polygon. The camera stays open."""
        with self._infer:
            self._zones = []
            self._monitoring = False
            self._board = StatusBoard()
            if self._pipeline is not None:
                self._pipeline.set_zones([])
        if self._zones_path.exists():
            self._zones_path.unlink()

    def publish_frame(self, frame: np.ndarray) -> None:
        """Draw one frame and, while monitoring, run detection on it.

        Args:
            frame: BGR image shaped ``(H, W, 3)``.
        """
        with self._infer:
            monitoring = self._monitoring
            pipeline = self._pipeline
            zones = list(self._zones)
            started = self._t0
            board = self._board
        shown = frame
        updated: StatusBoard | None = None
        if monitoring and pipeline is not None:
            result = pipeline.process_frame(frame, self._clock() - started)
            if result.annotated_frame is not None:
                shown = result.annotated_frame
            updated = refresh_board(
                board,
                result.workers,
                zones,
                now=self._clock(),
                interval=float(self._settings["alerts"].get("live_status_seconds", 5.0)),
                labels=self._settings["classes"].get("display_names", {}),
                wall_time=time.time(),
            )
        elif zones:
            color = _bgr(self._settings["visualize"]["colors"]["zone"])
            shown = draw_zones(frame, zones, color, thickness=2)
        encoded = _jpeg(shown)
        with self._lock:
            self._jpeg = encoded
        if updated is not None:
            with self._infer:
                if self._monitoring:
                    self._board = updated

    def latest_jpeg(self) -> bytes | None:
        """Return the newest JPEG, or ``None`` before the first frame."""
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict[str, Any]:
        """Return the camera, zone, and safety panel for the page.

        Returns:
            ``camera_open``, ``monitoring``, ``error``, ``zone``, and
            ``lines``.
        """
        with self._lock:
            capture = self._capture
            error = self._error
            board = self._board
        with self._infer:
            zones = list(self._zones)
            monitoring = self._monitoring
        zone = zones[0] if zones else None
        return {
            "camera_open": capture is not None and bool(capture.isOpened()),
            "monitoring": monitoring,
            "error": error,
            "source": self._source,
            "zone": None
            if zone is None
            else {
                "id": zone.zone_id,
                "name": zone.name,
                "required_ppe": list(zone.required_ppe),
                "points": [{"x": point.x, "y": point.y} for point in zone.polygon],
            },
            "lines": [
                {
                    "kind": line.kind,
                    "message": line.message,
                    "updated_at": line.updated_at,
                    "track_id": line.track_id,
                    "zone_name": line.zone_name,
                    "required_ppe": list(line.required_ppe),
                    "detected_ppe": list(line.detected_ppe),
                    "missing_ppe": list(line.missing_ppe),
                    "uncertain_ppe": list(line.uncertain_ppe),
                    "event_id": line.event_id,
                }
                for line in board.lines
            ],
        }

    def _loop(self) -> None:
        """Read until stop or the camera disconnects."""
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                message = getattr(capture, "open_error", None) or "The camera disconnected."
                with self._lock:
                    self._error = str(message)
                break
            try:
                self.publish_frame(frame)
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                break

    def _build_pipeline(self, zones: list[Zone]) -> SmartVisionPipeline:
        """Create the detection pipeline once, then reuse it."""
        if self._pipeline_factory is not None:
            pipeline = self._pipeline_factory(zones)
        else:
            from smartvision.detector import PPEDetector

            detector = PPEDetector(self._settings, project_root=self._root)
            pipeline = SmartVisionPipeline(self._settings, zones, detector, project_root=self._root)
        pipeline.violations.set_single_open_alert(True)
        return pipeline

    def _open(self, source: str) -> Any:
        """Open the capture with the injected hook or the project opener."""
        if self._open_capture is not None:
            return self._open_capture(source)
        from smartvision.pipeline import open_video_capture

        return open_video_capture(source)

    def _stop_thread(self) -> None:
        """Join the reader and release the device."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.release()
        self._stop.clear()


def _zone_from_request(
    name: str,
    points: list[tuple[float, float]],
    ppe: list[str],
    settings: dict[str, Any],
) -> Zone:
    """Validate a drawn polygon and build one zone."""
    label = name.strip()
    if not label:
        raise ValueError("Enter a zone name.")
    if len(points) < 3:
        raise ValueError("A zone needs at least 3 points.")
    polygon: list[Point] = []
    for x_pos, y_pos in points:
        x_value = float(x_pos)
        y_value = float(y_pos)
        if not 0.0 <= x_value <= 1.0 or not 0.0 <= y_value <= 1.0:
            raise ValueError("Zone points must stay inside the picture.")
        polygon.append(Point(x_value, y_value))
    allowed = {str(item) for item in settings["classes"]["ppe"]}
    required = tuple(str(item) for item in ppe)
    unknown = [item for item in required if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown PPE: {', '.join(unknown)}")
    return Zone(
        zone_id=_slug(label),
        name=label,
        polygon=tuple(polygon),
        required_ppe=required,
        risk_level=1,
    )


def _jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame, shrinking wide images so the page stays light."""
    height, width = frame.shape[:2]
    shown = frame
    if width > 960:
        scale = 960 / width
        shown = cv2.resize(frame, (960, max(1, int(round(height * scale)))))
    ok, encoded = cv2.imencode(".jpg", shown, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return b""
    return encoded.tobytes()


def _bgr(values: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    """Return a color as a BGR triple."""
    return (int(values[0]), int(values[1]), int(values[2]))


def _slug(name: str) -> str:
    """Turn a zone name into a stable id."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "zone"
