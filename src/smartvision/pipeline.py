"""Frame pipeline that connects detection, zones, compliance, tracking, and alerts.

``process_frame`` takes one numpy image and a timestamp. ``process_image``
reports violations on a still file without waiting for the hold timer.
``process_video`` yields frames from a file, an RTSP URL, or a webcam
index, skipping frames when configured. No stage opens a camera unless
``process_video`` is asked to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import yaml

from smartvision.alerts import Alert, Observation, ViolationManager
from smartvision.compliance import ComplianceEngine, classes_on_person, reconcile_worn
from smartvision.detector import Detection, PPEDetector
from smartvision.tracker import IoUTracker
from smartvision.visualize import FrameVisualizer
from smartvision.zones import Zone, find_zone_for_box, load_zones

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WorkerStatus(str, Enum):
    """Overlay state for one worker on one frame."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Worker:
    """One tracked person after compliance and the violation timer.

    Attributes:
        track_id: Stable worker id.
        bbox_xyxy: Person box as ``(x1, y1, x2, y2)`` in pixels.
        zone: Containing zone, or ``None`` when outside every zone.
        missing_ppe: Required classes confirmed absent.
        worn_ppe: Required classes detected on this person. Unselected
            classes are omitted even when the model sees them.
        status: ``OK``, ``WARNING``, or ``CRITICAL``.
        is_compliant: ``True`` when nothing required is confirmed missing.
        uncertain_ppe: Required classes whose body region is too cropped
            to confirm.
        required_ppe: Checklist that was applied to this person.
        gear: PPE boxes assigned to this person. Used to name each item
            when several workers share one photo.
        seen_ppe: Required classes whose boxes overlap this person, even
            when the body-band check did not accept them.
    """

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    zone: Zone | None
    missing_ppe: tuple[str, ...]
    worn_ppe: tuple[str, ...]
    status: WorkerStatus
    is_compliant: bool
    uncertain_ppe: tuple[str, ...] = ()
    required_ppe: tuple[str, ...] = ()
    gear: tuple[Detection, ...] = ()
    seen_ppe: tuple[str, ...] = ()


@dataclass
class FrameResult:
    """Everything one frame produces for the overlay and the caller.

    Attributes:
        detections: Person and PPE boxes from the detector.
        workers: Tracked people with zone, missing PPE, and status.
        alerts: Critical alerts raised on this frame.
        kpis: ``workers_in_zones``, ``active_critical_violations``, and
            ``zone_counts``.
        annotated_frame: Overlay image.
        timestamp: Clock value used for the violation timer.
    """

    detections: list[Detection] = field(default_factory=list)
    workers: list[Worker] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    kpis: dict[str, Any] = field(default_factory=dict)
    annotated_frame: np.ndarray | None = None
    timestamp: float = 0.0


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/settings.yaml``.

    Args:
        path: Settings file. The default is the repository copy.

    Returns:
        Parsed mapping.

    Raises:
        ValueError: The file did not contain a mapping.
    """
    source = Path(path) if path is not None else PROJECT_ROOT / "config" / "settings.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} did not contain a mapping")
    return payload


def open_video_capture(source: str | int) -> cv2.VideoCapture | Any:
    """Open a file, an RTSP URL, or a webcam index.

    Digit strings and integers are webcams. User index 0 is the iPhone
    camera and user index 1 is the built-in laptop camera.

    Args:
        source: Path, URL, or webcam index.

    Returns:
        A capture with ``read``, ``get``, and ``release``. The caller
        releases it.

    Raises:
        RuntimeError: A webcam index has no matching camera.
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        from smartvision.cameras import open_webcam

        return open_webcam(int(source))
    return cv2.VideoCapture(source)


def is_live_source(source: str | int) -> bool:
    """Report whether ``source`` is a webcam or a network stream.

    Args:
        source: Value passed to :func:`open_video_capture`.

    Returns:
        ``True`` for an integer device, a digit string, or a URL.
    """
    if isinstance(source, int):
        return True
    text = str(source)
    return text.isdigit() or "://" in text


class SmartVisionPipeline:
    """Hold the stage objects and run frames through them."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        zones: list[Zone] | None = None,
        detector: PPEDetector | Any | None = None,
        *,
        project_root: Path | None = None,
        clock: Callable[[], float] | None = None,
        show_zones: bool = True,
    ) -> None:
        """Build the stages from config.

        Args:
            settings: Parsed settings. Loaded from disk when omitted.
            zones: Danger zones. Loaded from ``config/zones.yaml`` when
                omitted.
            detector: Object with ``detect(frame)``. The real
                :class:`~smartvision.detector.PPEDetector` is built when
                this is omitted.
            project_root: Base for relative config and output paths.
            clock: Clock forwarded to the violation manager.
            show_zones: When false, compliance still uses the zones but the
                overlay leaves the polygon off. Photos and recorded video
                pass false. The live camera leaves this true.
        """
        self._root = project_root or PROJECT_ROOT
        self._settings = settings if settings is not None else load_settings()
        self._zones = zones if zones is not None else load_zones(self._root / "config" / "zones.yaml")
        self.detector = detector if detector is not None else PPEDetector(self._settings, project_root=self._root)
        self.tracker = IoUTracker(self._settings)
        self.compliance = ComplianceEngine(self._settings)
        self.violations = ViolationManager(
            self._settings,
            clock=clock,
            project_root=self._root,
            draw_zones=show_zones,
        )
        self.visualizer = FrameVisualizer(self._settings)
        self._show_zones = show_zones
        self._frame_skip = int(self._settings["pipeline"]["frame_skip"])
        self._frame_index = 0
        self._clock = clock or time.monotonic
        self._person_name = str(self._settings["classes"]["person"])

    def set_zones(self, zones: list[Zone]) -> None:
        """Replace the zones used on later frames.

        The detector, tracker, and open camera stay as they are. A webcam
        session uses this when the operator saves a new polygon.

        Args:
            zones: Zones for the next frame. An empty list means nobody is
                inside a zone.
        """
        self._zones = list(zones)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> FrameResult:
        """Run one frame through detection, zones, compliance, tracking, and alerts.

        Args:
            frame: BGR image shaped ``(H, W, 3)``. A camera is not required.
            timestamp: Timer clock for this frame. When omitted, the
                pipeline clock is read.

        Returns:
            Detections, workers, new critical alerts, KPIs, and the overlay.
        """
        moment = self._clock() if timestamp is None else float(timestamp)
        return self._process(frame, moment, immediate=False)

    def process_image(self, path: str | Path) -> FrameResult:
        """Check one image and report violations without the hold timer.

        Args:
            path: Image file readable by OpenCV.

        Returns:
            The same result shape as :meth:`process_frame`. A person inside
            a zone with missing PPE is critical on this single frame.

        Raises:
            FileNotFoundError: The image could not be read.
        """
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return self._process(frame, timestamp=0.0, immediate=True)

    def process_video(
        self,
        source: str | int,
        frame_skip: int | None = None,
        capture: cv2.VideoCapture | Any | None = None,
    ) -> Iterator[FrameResult]:
        """Yield a result for each processed frame of a video source.

        ``frame_skip`` is the number of frames left unread by the model
        between processed frames. ``0`` processes every frame. File
        timestamps advance by ``frame_index / fps``. Webcam and RTSP
        timestamps use the pipeline clock.

        Args:
            source: File path, RTSP URL, or webcam index.
            frame_skip: Overrides ``pipeline.frame_skip`` when set.
            capture: Open capture to read. When omitted, ``source`` is
                opened and closed here.

        Yields:
            One :class:`FrameResult` per processed frame.

        Raises:
            ValueError: ``frame_skip`` is negative.
        """
        skip = self._frame_skip if frame_skip is None else int(frame_skip)
        if skip < 0:
            raise ValueError("frame_skip must be >= 0")
        owns_capture = capture is None
        cap = capture if capture is not None else open_video_capture(source)
        live = is_live_source(source)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 25.0
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if index % (skip + 1) != 0:
                    index += 1
                    continue
                timestamp = self._clock() if live else index / fps
                yield self.process_frame(frame, timestamp)
                index += 1
        finally:
            if owns_capture:
                cap.release()

    def _process(self, frame: np.ndarray, timestamp: float, immediate: bool) -> FrameResult:
        """Track, judge, and draw one frame."""
        detections = list(self.detector.detect(frame))
        people = [det for det in detections if det.class_name == self._person_name]
        gear = [det for det in detections if det.class_name != self._person_name]
        tracks = self.tracker.update(self._frame_index, people)
        self._frame_index += 1
        height, width = frame.shape[:2]
        tracked_people = [
            Detection(self._person_name, 1.0, track.bbox_xyxy)
            for track in tracks
        ]
        zones = [
            find_zone_for_box(track.bbox_xyxy, self._zones, (width, height))
            for track in tracks
        ]
        required = [self.compliance.required_for(zone) for zone in zones]
        judged = self.compliance.evaluate(
            tracked_people,
            gear,
            required,
            image_size=(width, height),
        )
        reconciled = [
            reconcile_worn(
                result.required_ppe,
                result.worn,
                result.missing,
                classes_on_person(track.bbox_xyxy, gear, result.required_ppe),
            )
            for track, result in zip(tracks, judged)
        ]
        observations = [
            Observation(
                track_id=track.track_id,
                bbox_xyxy=track.bbox_xyxy,
                zone=zone,
                missing_ppe=missing,
                is_compliant=compliant,
            )
            for track, zone, (_, missing, compliant, _) in zip(tracks, zones, reconciled)
        ]
        statuses, alerts = self.violations.update(
            frame,
            observations,
            now=timestamp,
            immediate=immediate,
        )
        workers = [
            Worker(
                track_id=track.track_id,
                bbox_xyxy=track.bbox_xyxy,
                zone=zone,
                missing_ppe=missing,
                worn_ppe=worn,
                status=WorkerStatus(statuses[track.track_id]),
                is_compliant=compliant,
                uncertain_ppe=result.uncertain,
                required_ppe=result.required_ppe,
                gear=result.assigned,
                seen_ppe=seen,
            )
            for track, zone, result, (worn, missing, compliant, seen) in zip(
                tracks, zones, judged, reconciled
            )
        ]
        annotated = self.visualizer.draw(
            frame,
            self._zones,
            workers,
            detections,
            show_zones=self._show_zones,
        )
        return FrameResult(
            detections=detections,
            workers=workers,
            alerts=alerts,
            kpis=_kpis(workers),
            annotated_frame=annotated,
            timestamp=timestamp,
        )


def _kpis(workers: list[Worker]) -> dict[str, Any]:
    """Count workers inside zones, critical workers, and workers per zone."""
    zone_counts: dict[str, int] = {}
    in_zones = 0
    critical = 0
    for worker in workers:
        if worker.status is WorkerStatus.CRITICAL:
            critical += 1
        if worker.zone is None:
            continue
        in_zones += 1
        zone_counts[worker.zone.name] = zone_counts.get(worker.zone.name, 0) + 1
    return {
        "workers_in_zones": in_zones,
        "active_critical_violations": critical,
        "zone_counts": zone_counts,
    }


Pipeline = SmartVisionPipeline
