"""SmartVision dashboard.

Run from the repository root:

    streamlit run app/streamlit_app.py

The page calls :class:`smartvision.pipeline.SmartVisionPipeline`. YOLO weights
are loaded once. Sidebar sliders scale the confidence cutoffs, the
persistence timer, and how many frames are skipped between inferences.
"""

from __future__ import annotations

import base64
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import smartvision  # noqa: E402
from scripts.zone_helper import grab_frame, save_zones_file, unique_id  # noqa: E402
from smartvision.alerts import Alert, ViolationManager  # noqa: E402
from smartvision.detector import PPEDetector  # noqa: E402
from smartvision.live_status import StatusBoard, refresh_board  # noqa: E402
from smartvision.pipeline import (  # noqa: E402
    FrameResult,
    SmartVisionPipeline,
    Worker,
    is_live_source,
    load_settings,
    open_video_capture,
)
from smartvision.zones import Point, Zone, draw_zones, load_zones, save_zones  # noqa: E402

SETTINGS = ROOT / "config" / "settings.yaml"
ZONES = ROOT / "config" / "zones.yaml"
WEBCAM_ZONES = ROOT / "outputs" / "webcam_zone.yaml"
SAMPLES = ROOT / "data" / "samples"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAGE_LIVE = "Live / Video"
PAGE_IMAGE = "Image"
PAGE_ZONES = "Zone editor"
PAGE_ALERTS = "Alerts history"
STATUS_COLOR = {"OK": "#1b7f3a", "WARNING": "#b86e00", "CRITICAL": "#c0392b"}
_POLYGON_DRAW = components.declare_component("polygon_draw", path=str(ROOT / "app" / "polygon_draw"))


def main() -> None:
    """Render the sidebar and the selected page."""
    st.set_page_config(page_title="SmartVision", layout="wide")
    _style()
    _init_state()
    settings = load_settings(SETTINGS)
    confidence, hold, frame_skip = _sidebar(settings)
    st.title("SmartVision")
    st.caption(f"PPE compliance and danger-zone monitoring (v{smartvision.__version__}).")
    page = st.session_state.page
    if page == PAGE_LIVE:
        _page_live(settings, confidence, hold, frame_skip)
    elif page == PAGE_IMAGE:
        _page_image(settings, confidence, hold, frame_skip)
    elif page == PAGE_ZONES:
        _page_zones(settings)
    elif page == PAGE_ALERTS:
        _page_alerts()
    else:
        _page_live(settings, confidence, hold, frame_skip)


def _style() -> None:
    """Apply the green / amber / red status colours."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; }
        div[data-testid="stMetric"] {
            background: #f7f5f2;
            border: 1px solid #e4ddd4;
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    """Create session keys used across reruns."""
    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("capture", None)
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("live_source", None)
    st.session_state.setdefault("frame_index", 0)
    st.session_state.setdefault("fps", 25.0)
    st.session_state.setdefault("live_t0", 0.0)
    st.session_state.setdefault("last_rgb", None)
    st.session_state.setdefault("last_kpis", None)
    st.session_state.setdefault("last_workers", None)
    st.session_state.setdefault("editor_zones", None)
    st.session_state.setdefault("draft_points", [])
    st.session_state.setdefault("editor_frame", None)
    st.session_state.setdefault("image_result", None)
    st.session_state.setdefault("live_zones", [])
    st.session_state.setdefault("live_draft", [])
    st.session_state.setdefault("previewing", False)
    st.session_state.setdefault("drawing", False)
    st.session_state.setdefault("draw_frame", None)
    st.session_state.setdefault("preview_bgr", None)
    st.session_state.setdefault("polygon_epoch", 0)
    st.session_state.setdefault("webcam_paused", False)
    st.session_state.setdefault("live_status_board", None)
    st.session_state.setdefault("last_source_kind", None)
    st.session_state.setdefault("live_confidence", 0.25)
    st.session_state.setdefault("live_hold", 2.0)
    st.session_state.setdefault("live_frame_skip", 0)


def _sidebar(settings: dict) -> tuple[float, float, int]:
    """Draw navigation and the three runtime sliders.

    Args:
        settings: Parsed ``config/settings.yaml``.

    Returns:
        Confidence, persistence seconds, and frame skip.
    """
    st.sidebar.header("SmartVision")
    st.session_state.page = st.sidebar.radio(
        "Page",
        [PAGE_LIVE, PAGE_IMAGE, PAGE_ZONES, PAGE_ALERTS],
    )
    st.sidebar.header("Run controls")
    default_conf = float(settings["model"]["confidence"])
    confidence = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_conf,
        step=0.05,
        help="Scales every cutoff in settings.yaml. The default matches the file.",
    )
    scaled = scaled_model_thresholds(settings, confidence)
    st.sidebar.caption(
        f"Helmet {scaled['class_confidence']['helmet']:.2f} · "
        f"gloves {scaled['class_confidence']['gloves']:.2f} · "
        f"person {scaled['person_conf']:.2f}"
    )
    hold = st.sidebar.slider(
        "Persistence seconds",
        min_value=0.0,
        max_value=30.0,
        value=float(settings["compliance"]["violation_hold_seconds"]),
        step=0.5,
        help="How long missing PPE inside a zone must persist before a critical alert.",
    )
    frame_skip = st.sidebar.slider(
        "Frame skip",
        min_value=0,
        max_value=30,
        value=3,
        step=1,
        help="Frames left unprocessed between inferences. 3 processes every 4th frame.",
    )
    return float(confidence), float(hold), int(frame_skip)


def scaled_model_thresholds(settings: dict, confidence: float) -> dict:
    """Scale the configured cutoffs so the slider default matches the file.

    Args:
        settings: Parsed settings.
        confidence: Sidebar confidence value.

    Returns:
        ``confidence``, ``person_conf``, and ``class_confidence`` for
        :meth:`PPEDetector.set_thresholds`.
    """
    model = settings["model"]
    base = float(model["confidence"])
    scale = 1.0 if base <= 0 else float(confidence) / base
    return {
        "confidence": float(confidence),
        "person_conf": min(0.99, float(model["person_conf"]) * scale),
        "class_confidence": {
            str(name): min(0.99, float(value) * scale)
            for name, value in model["class_confidence"].items()
        },
    }


def runtime_settings(hold: float, frame_skip: int) -> dict:
    """Return settings with the sidebar hold time and frame skip applied.

    Args:
        hold: Persistence seconds.
        frame_skip: Frames skipped between inferences.

    Returns:
        A freshly loaded settings mapping.
    """
    settings = load_settings(SETTINGS)
    settings["compliance"]["violation_hold_seconds"] = float(hold)
    settings["pipeline"]["frame_skip"] = int(frame_skip)
    return settings


@st.cache_resource(show_spinner="Loading YOLO models…")
def cached_detector() -> PPEDetector:
    """Load both YOLO models once for this Streamlit process."""
    return PPEDetector(load_settings(SETTINGS), project_root=ROOT)


def _apply_confidence(settings: dict, confidence: float) -> None:
    """Push the sidebar confidence onto the cached detector."""
    cached_detector().set_thresholds(**scaled_model_thresholds(settings, confidence))


def _page_live(settings: dict, confidence: float, hold: float, frame_skip: int) -> None:
    """Play a video, stream, or webcam through the pipeline one frame at a time."""
    st.subheader("Live / Video")
    source = _live_source_form()
    if st.session_state.get("live_source_kind") == "Webcam":
        if st.session_state.live_running or st.session_state.capture is not None:
            _halt_live()
        return
    webcam = _is_webcam(source)
    kind = "webcam" if webcam else "file"
    previous = st.session_state.last_source_kind
    if previous != kind:
        if previous == "webcam":
            _halt_live()
        st.session_state.last_source_kind = kind
        if webcam:
            st.session_state.webcam_paused = False
    if webcam:
        _page_webcam(settings, confidence, hold, frame_skip, source)
        return
    start, stop = st.columns(2)
    if start.button("Start", type="primary"):
        if not source:
            st.error("Choose a sample, upload a video, or enter an RTSP URL.")
        else:
            _begin_live(source, hold, frame_skip, webcam)
    if stop.button("Stop", disabled=not st.session_state.live_running):
        _halt_live()
        st.info("Stopped.")
    _show_live_output()
    if st.session_state.live_running:
        _apply_confidence(settings, confidence)
        pipeline = st.session_state.pipeline
        if pipeline is not None:
            pipeline.violations.set_hold_seconds(hold)
        finished = _advance_live_frame(frame_skip)
        if finished:
            st.rerun()
    st.subheader("Recent alerts")
    _render_alert_cards(_recent_alerts(8))


def _page_webcam(
    settings: dict,
    confidence: float,
    hold: float,
    frame_skip: int,
    source: str | None,
) -> None:
    """Keep the webcam open while the operator draws a zone and while detection runs."""
    st.session_state.live_confidence = float(confidence)
    st.session_state.live_hold = float(hold)
    st.session_state.live_frame_skip = int(frame_skip)
    if not source:
        st.error("Choose a webcam.")
        return
    if st.session_state.live_source not in (None, source):
        _release_capture_only()
        st.session_state.live_zones = []
        st.session_state.live_running = False
        st.session_state.live_draft = []
        st.session_state.live_status_board = None
    st.session_state.live_source = source
    if st.session_state.webcam_paused:
        st.info("Camera is off.")
        if st.button("Start camera", type="primary"):
            st.session_state.webcam_paused = False
            st.rerun()
        return
    _webcam_live_fragment()
    if st.session_state.live_running:
        edit, reset = st.columns(2)
        if edit.button("Edit zone"):
            st.session_state.live_running = False
            st.session_state.live_draft = []
            st.session_state.polygon_epoch += 1
            st.rerun()
        if reset.button("Reset zone"):
            _clear_webcam_zone()
            st.rerun()
    else:
        st.caption(
            "Click each corner on the live picture, choose the PPE this zone requires, and save it. "
            "Detection starts when the zone is saved. The camera stays on."
        )
        _live_zone_form(settings)
    if st.button("Stop camera"):
        _halt_live()
        st.session_state.webcam_paused = True
        st.rerun()


@st.fragment(run_every=0.2)
def _webcam_live_fragment() -> None:
    """Read the open camera. Saving a zone does not close it."""
    source = st.session_state.get("live_source")
    if not source or st.session_state.get("webcam_paused"):
        return
    if not _ensure_webcam_capture(str(source)):
        return
    capture = st.session_state.capture
    frame_skip = int(st.session_state.get("live_frame_skip") or 0)
    if int(st.session_state.frame_index) > 0:
        for _ in range(frame_skip):
            ok, _skipped = capture.read()
            if not ok:
                _fail_webcam_read(capture)
                return
            st.session_state.frame_index += 1
    ok, frame = capture.read()
    if not ok or frame is None:
        _fail_webcam_read(capture)
        return
    st.session_state.frame_index += 1
    st.session_state.preview_bgr = frame.copy()
    if st.session_state.live_running and st.session_state.pipeline is not None:
        _monitor_webcam_frame(frame)
        return
    _draw_on_live_frame()


def _ensure_webcam_capture(source: str) -> bool:
    """Open the selected webcam once and keep that capture."""
    capture = st.session_state.capture
    if capture is not None and st.session_state.live_source == source and capture.isOpened():
        return True
    if capture is not None:
        capture.release()
        st.session_state.capture = None
    try:
        capture = open_video_capture(source)
    except RuntimeError as exc:
        st.error(str(exc))
        st.session_state.webcam_paused = True
        st.rerun()
        return False
    if not capture.isOpened():
        st.error(getattr(capture, "open_error", None) or f"Could not open camera {source}")
        capture.release()
        st.session_state.webcam_paused = True
        st.rerun()
        return False
    st.session_state.capture = capture
    st.session_state.live_source = source
    if not st.session_state.live_t0:
        st.session_state.live_t0 = time.monotonic()
    return True


def _fail_webcam_read(capture: object) -> None:
    """Stop the preview after the camera disconnects."""
    message = getattr(capture, "open_error", None) or "The camera disconnected."
    st.error(str(message))
    _release_capture_only()
    st.session_state.webcam_paused = True
    st.rerun()


def _draw_on_live_frame() -> None:
    """Show the polygon drawer on the current camera frame."""
    settings = load_settings(SETTINGS)
    frame = st.session_state.preview_bgr
    color = tuple(int(value) for value in settings["visualize"]["colors"]["zone"])
    shown = draw_zones(frame, list(st.session_state.live_zones), color, thickness=2)
    clicked = _POLYGON_DRAW(
        image=_jpeg_base64(shown),
        points=[{"x": x_pos, "y": y_pos} for x_pos, y_pos in st.session_state.live_draft],
        key=f"live_polygon_draw_{st.session_state.polygon_epoch}",
        default=None,
    )
    parsed = _clicks_to_points(clicked)
    if parsed is not None:
        st.session_state.live_draft = parsed


def _monitor_webcam_frame(frame: np.ndarray) -> None:
    """Detect people and PPE on one webcam frame and refresh the safety panel."""
    pipeline = st.session_state.pipeline
    settings = load_settings(SETTINGS)
    _apply_confidence(settings, float(st.session_state.live_confidence))
    pipeline.violations.set_hold_seconds(float(st.session_state.live_hold))
    timestamp = time.monotonic() - float(st.session_state.live_t0 or time.monotonic())
    try:
        result = pipeline.process_frame(frame, timestamp)
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        return
    _remember_result(result)
    _refresh_live_board(result.workers, settings)
    if st.session_state.last_rgb is not None:
        st.image(st.session_state.last_rgb, use_container_width=True)
    _render_live_safety()


def _refresh_live_board(workers: list[Worker], settings: dict) -> None:
    """Rebuild the safety panel when five seconds have passed."""
    interval = float(settings["alerts"].get("live_status_seconds", 5.0))
    labels = settings["classes"].get("display_names", {})
    board = st.session_state.live_status_board or StatusBoard()
    now = time.monotonic()
    st.session_state.live_status_board = refresh_board(
        board,
        workers,
        list(st.session_state.live_zones),
        now=now,
        interval=interval,
        labels=labels,
        wall_time=time.time(),
    )


def _render_live_safety() -> None:
    """Show the latest safety lines directly under the camera image."""
    st.subheader("Live Safety Alerts")
    board = st.session_state.live_status_board or StatusBoard()
    if not board.lines:
        st.caption("The first safety check appears with the first processed frame.")
        return
    for line in board.lines:
        stamp = datetime.fromtimestamp(line.updated_at).strftime("%H:%M:%S")
        if line.kind == "violation":
            st.error(line.message)
        elif line.kind == "uncertain":
            st.warning(line.message)
        elif line.kind == "compliant":
            st.success(line.message)
        else:
            st.info(line.message)
        st.caption(f"Latest evaluation: {stamp}")


def _is_webcam(source: str | None) -> bool:
    """Report whether ``source`` is a webcam index."""
    return bool(source) and str(source).isdigit()


def _clear_webcam_zone() -> None:
    """Drop the webcam polygon and return to drawing. The camera stays open."""
    st.session_state.live_zones = []
    st.session_state.live_running = False
    st.session_state.live_draft = []
    st.session_state.live_status_board = None
    st.session_state.polygon_epoch += 1
    pipeline = st.session_state.get("pipeline")
    if pipeline is not None:
        pipeline.set_zones([])
    if WEBCAM_ZONES.exists():
        WEBCAM_ZONES.unlink()


def _jpeg_base64(frame: np.ndarray) -> str:
    """Encode a BGR frame as a JPEG data payload for the drawing component."""
    height, width = frame.shape[:2]
    shown = frame
    if width > 960:
        scale = 960 / width
        shown = cv2.resize(frame, (960, max(1, int(round(height * scale)))))
    ok, encoded = cv2.imencode(".jpg", shown, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _clicks_to_points(value: object) -> list[tuple[float, float]] | None:
    """Read normalized corners sent by the drawing component.

    Args:
        value: Component payload, or ``None`` before the first click.

    Returns:
        Corners inside the unit square, or ``None`` before the drawer reports a click.
    """
    if not isinstance(value, list):
        return None
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        try:
            x_pos = float(item["x"])
            y_pos = float(item["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0.0 <= x_pos <= 1.0 or not 0.0 <= y_pos <= 1.0:
            return None
        points.append((x_pos, y_pos))
    return points


def _live_zone_form(settings: dict) -> None:
    """Name the polygon and start detection without closing the camera."""
    st.markdown("**Draw polygon zone**")
    labels = settings["classes"].get("display_names", {})
    name = st.text_input("Zone name", placeholder="Machine 1", key="live_zone_name")
    default_ppe = [item for item in settings["compliance"]["default_required_ppe"] if item in settings["classes"]["ppe"]]
    ppe = st.multiselect(
        "Required PPE",
        list(settings["classes"]["ppe"]),
        default=default_ppe,
        format_func=lambda item: str(labels.get(item, item)),
        key="live_zone_ppe",
    )
    if st.button("Save zone", key="live_add_zone"):
        points = list(st.session_state.live_draft)
        if len(points) < 3:
            st.error("A zone needs at least 3 points.")
            return
        if not name.strip():
            st.error("Enter a zone name.")
            return
        zone = Zone(
            zone_id=unique_id(name.strip(), set()),
            name=name.strip(),
            polygon=tuple(Point(x_pos, y_pos) for x_pos, y_pos in points),
            required_ppe=tuple(ppe),
            risk_level=1,
        )
        st.session_state.live_zones = [zone]
        save_zones(WEBCAM_ZONES, [zone])
        if not _activate_webcam_monitoring():
            return
        st.rerun()


def _activate_webcam_monitoring() -> bool:
    """Start PPE checks on the camera that is already open.

    Returns:
        ``False`` when the model weights cannot be loaded.
    """
    hold = float(st.session_state.live_hold)
    frame_skip = int(st.session_state.live_frame_skip)
    zones = list(st.session_state.live_zones)
    try:
        detector = cached_detector()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return False
    _apply_confidence(load_settings(SETTINGS), float(st.session_state.live_confidence))
    pipeline = st.session_state.pipeline
    if pipeline is None:
        settings = runtime_settings(hold, frame_skip)
        pipeline = SmartVisionPipeline(settings, zones, detector, project_root=ROOT)
        st.session_state.pipeline = pipeline
        st.session_state.live_t0 = time.monotonic()
        st.session_state.frame_index = 0
    else:
        pipeline.set_zones(zones)
        pipeline.violations.set_hold_seconds(hold)
    pipeline.violations.set_single_open_alert(True)
    st.session_state.live_running = True
    st.session_state.drawing = False
    st.session_state.previewing = True
    st.session_state.live_draft = []
    st.session_state.polygon_epoch += 1
    st.session_state.live_status_board = StatusBoard()
    return True


def _live_source_form() -> str | None:
    """Return the source chosen in the form, or ``None`` when it is incomplete."""
    kind = st.radio(
        "Source",
        ["Sample", "Upload", "RTSP", "Webcam"],
        horizontal=True,
        key="live_source_kind",
    )
    if kind == "Sample":
        videos = _list_media(VIDEO_SUFFIXES)
        if not videos:
            st.warning("No videos in data/samples.")
            return None
        choice = st.selectbox("Video", videos, format_func=lambda path: path.name)
        return str(choice)
    if kind == "Upload":
        upload = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
        if upload is None:
            return None
        return str(_store_upload(upload))
    if kind == "RTSP":
        url = st.text_input("RTSP URL", placeholder="rtsp://camera/stream")
        return url.strip() or None
    st.info("The live camera runs on its own page so the picture stays up while you draw a zone.")
    st.link_button("Open live camera", "http://localhost:8000")
    return None


def _begin_live(source: str, hold: float, frame_skip: int, webcam: bool) -> None:
    """Reset tracker state and mark the live loop as running.

    A webcam run uses the zones drawn on the preview. Other sources use
    ``config/zones.yaml``.
    """
    zones = list(st.session_state.live_zones) if webcam else load_zones(ZONES)
    _release_live_resources()
    try:
        detector = cached_detector()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    settings = runtime_settings(hold, frame_skip)
    pipeline = SmartVisionPipeline(settings, zones, detector, project_root=ROOT)
    st.session_state.pipeline = pipeline
    st.session_state.live_source = source
    st.session_state.live_running = True
    st.session_state.previewing = False
    st.session_state.drawing = False
    st.session_state.capture = None
    st.session_state.frame_index = 0
    st.session_state.live_t0 = time.monotonic()
    st.session_state.last_rgb = None
    st.session_state.last_kpis = None
    st.session_state.last_workers = None


def _release_capture_only() -> None:
    """Close the camera device and leave the pipeline in place."""
    capture = st.session_state.get("capture")
    if capture is not None:
        capture.release()
    st.session_state.capture = None


def _release_live_resources() -> None:
    """Release the capture and close the pipeline database connection."""
    _release_capture_only()
    pipeline = st.session_state.get("pipeline")
    if pipeline is not None:
        pipeline.violations.close()
    st.session_state.pipeline = None


def _halt_live() -> None:
    """Stop the camera and drop the webcam zone."""
    _release_live_resources()
    st.session_state.previewing = False
    st.session_state.drawing = False
    st.session_state.draw_frame = None
    st.session_state.preview_bgr = None
    st.session_state.live_running = False
    st.session_state.last_rgb = None
    st.session_state.last_kpis = None
    st.session_state.last_workers = None
    st.session_state.live_zones = []
    st.session_state.live_draft = []
    st.session_state.live_status_board = None
    if WEBCAM_ZONES.exists():
        WEBCAM_ZONES.unlink()


def _end_live_read(source: str, capture: object) -> None:
    """Stop after a read fails. A dead webcam reports its open error."""
    message = getattr(capture, "open_error", None)
    if is_live_source(source) and message:
        st.error(str(message))
    else:
        st.success("Video finished.")
    _halt_live()


def _advance_live_frame(frame_skip: int) -> bool:
    """Process the next kept frame.

    Args:
        frame_skip: Frames to drop between inferences.

    Returns:
        ``True`` when the UI should rerun to show the new frame.
    """
    pipeline = st.session_state.pipeline
    source = st.session_state.live_source
    if pipeline is None or not source:
        _halt_live()
        return False
    if st.session_state.capture is None:
        try:
            capture = open_video_capture(source)
        except RuntimeError as exc:
            st.error(str(exc))
            _halt_live()
            return False
        if not capture.isOpened():
            st.error(getattr(capture, "open_error", None) or f"Could not open {source}")
            capture.release()
            _halt_live()
            return False
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        st.session_state.capture = capture
        st.session_state.fps = fps if fps > 1e-3 else 25.0
        st.session_state.frame_index = 0
        st.session_state.live_t0 = time.monotonic()
    capture = st.session_state.capture
    if st.session_state.frame_index > 0:
        for _ in range(frame_skip):
            ok, _skipped = capture.read()
            if not ok:
                _end_live_read(source, capture)
                return False
            st.session_state.frame_index += 1
    ok, frame = capture.read()
    if not ok or frame is None:
        _end_live_read(source, capture)
        return False
    index = st.session_state.frame_index
    if is_live_source(source):
        timestamp = time.monotonic() - float(st.session_state.live_t0)
    else:
        timestamp = index / float(st.session_state.fps)
    st.session_state.frame_index = index + 1
    try:
        result = pipeline.process_frame(frame, timestamp)
    except Exception as exc:
        st.error(f"Processing stopped: {exc}")
        _halt_live()
        return False
    _remember_result(result)
    return True


def _show_live_output() -> None:
    """Draw the latest annotated frame, KPIs, and worker table."""
    kpis = st.session_state.last_kpis or {
        "workers_in_zones": 0,
        "active_critical_violations": 0,
        "zone_counts": {},
    }
    _kpi_row(kpis)
    if st.session_state.last_rgb is not None and not st.session_state.previewing:
        st.image(st.session_state.last_rgb, use_container_width=True)
    elif not st.session_state.live_running and not st.session_state.previewing:
        st.info("Start a video to see annotated frames.")
    workers = st.session_state.last_workers or []
    if workers:
        st.dataframe(_worker_table(workers), hide_index=True, use_container_width=True)


def _page_image(settings: dict, confidence: float, hold: float, frame_skip: int) -> None:
    """Run one still image without the persistence timer."""
    st.subheader("Image")
    kind = st.radio("Image source", ["Sample", "Upload"], horizontal=True)
    path: Path | None = None
    if kind == "Sample":
        images = _list_media(IMAGE_SUFFIXES)
        if not images:
            st.warning("No images in data/samples.")
        else:
            path = st.selectbox("Image", images, format_func=lambda item: item.name)
    else:
        upload = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if upload is not None:
            path = _store_upload(upload)
    if st.button("Check image", type="primary", disabled=path is None):
        try:
            _apply_confidence(settings, confidence)
            pipeline = SmartVisionPipeline(
                runtime_settings(hold, frame_skip),
                load_zones(ZONES),
                cached_detector(),
                project_root=ROOT,
            )
            try:
                st.session_state.image_result = pipeline.process_image(path)
            finally:
                pipeline.violations.close()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not check the image: {exc}")
    result = st.session_state.image_result
    if result is None:
        return
    if result.annotated_frame is not None:
        st.image(_bgr_to_rgb(result.annotated_frame), use_container_width=True)
    st.dataframe(_worker_table(result.workers), hide_index=True, use_container_width=True)
    if result.alerts:
        st.subheader("Violations on this image")
        _render_alert_cards(result.alerts)


def _page_zones(settings: dict) -> None:
    """Edit polygons on a reference frame and save ``config/zones.yaml``."""
    st.subheader("Zone editor")
    st.caption(
        "streamlit-drawable-canvas does not load on Streamlit "
        f"{st.__version__} (it needs a removed image helper), so points are "
        "entered here and drawn on the frame. Values are normalized: 0 is the "
        "left or top, 1 is the right or bottom."
    )
    if st.session_state.editor_zones is None:
        st.session_state.editor_zones = load_zones(ZONES) if ZONES.is_file() else []
    _load_reference_frame()
    _draw_editor_preview(settings)
    _draft_point_controls()
    _add_zone_form(settings)
    _save_zone_form()


def _load_reference_frame() -> None:
    """Store one BGR reference frame for the editor preview."""
    kind = st.radio("Reference", ["Sample image", "Sample video", "Upload"], horizontal=True, key="zone_source_kind")
    frame_index = 0
    source: Path | None = None
    if kind == "Sample image":
        images = _list_media(IMAGE_SUFFIXES)
        if images:
            source = st.selectbox("Image", images, format_func=lambda item: item.name, key="zone_image")
    elif kind == "Sample video":
        videos = _list_media(VIDEO_SUFFIXES)
        if videos:
            source = st.selectbox("Video", videos, format_func=lambda item: item.name, key="zone_video")
            frame_index = int(st.number_input("Frame number", min_value=0, value=0, step=1))
    else:
        upload = st.file_uploader(
            "Image or video",
            type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov"],
            key="zone_upload",
        )
        if upload is not None:
            source = _store_upload(upload)
            if source.suffix.lower() in VIDEO_SUFFIXES:
                frame_index = int(st.number_input("Frame number", min_value=0, value=0, step=1, key="zone_upload_frame"))
    if st.button("Load frame", disabled=source is None):
        try:
            st.session_state.editor_frame = grab_frame(source, frame_index)
        except (OSError, ValueError) as exc:
            st.error(str(exc))


def _draw_editor_preview(settings: dict) -> None:
    """Show saved zones and the polygon currently being drawn."""
    frame = st.session_state.editor_frame
    if frame is None:
        st.info("Load a reference frame, then add normalized points.")
        return
    color = tuple(int(value) for value in settings["visualize"]["colors"]["zone"])
    canvas = draw_zones(frame, list(st.session_state.editor_zones), color, thickness=2)
    height, width = canvas.shape[:2]
    pixels = [
        (int(round(x_pos * width)), int(round(y_pos * height)))
        for x_pos, y_pos in st.session_state.draft_points
    ]
    for index, point in enumerate(pixels):
        if index:
            cv2.line(canvas, pixels[index - 1], point, (0, 255, 0), 2)
        cv2.circle(canvas, point, 5, (0, 255, 0), -1)
    if len(pixels) >= 3:
        cv2.line(canvas, pixels[-1], pixels[0], (0, 255, 0), 1)
    st.image(_bgr_to_rgb(canvas), use_container_width=True)


def _draft_point_controls() -> None:
    """Add, paste, undo, or clear the open polygon."""
    left, right = st.columns(2)
    x_pos = left.slider("Point x", 0.0, 1.0, 0.5, 0.01)
    y_pos = right.slider("Point y", 0.0, 1.0, 0.5, 0.01)
    add, undo, clear = st.columns(3)
    if add.button("Add point"):
        st.session_state.draft_points.append((float(x_pos), float(y_pos)))
        st.rerun()
    if undo.button("Undo point"):
        if st.session_state.draft_points:
            st.session_state.draft_points.pop()
        st.rerun()
    if clear.button("Clear points"):
        st.session_state.draft_points = []
        st.rerun()
    pasted = st.text_area(
        "Or paste points, one `x, y` pair per line",
        placeholder="0.02, 0.20\n0.48, 0.20\n0.48, 0.98",
    )
    if st.button("Replace points from text"):
        try:
            st.session_state.draft_points = parse_point_text(pasted)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    if st.session_state.draft_points:
        st.write(
            "Current points: "
            + "; ".join(f"({x_pos:.2f}, {y_pos:.2f})" for x_pos, y_pos in st.session_state.draft_points)
        )


def _add_zone_form(settings: dict) -> None:
    """Turn the open polygon into a zone on the in-memory list."""
    st.markdown("**New zone**")
    name = st.text_input("Name", placeholder="Loading bay")
    risk = st.number_input("Risk level", min_value=0, max_value=100, value=1, step=1)
    ppe = st.multiselect("Required PPE", list(settings["classes"]["ppe"]))
    if st.button("Add zone"):
        points = st.session_state.draft_points
        if len(points) < 3:
            st.error("A zone needs at least 3 points.")
            return
        if not name.strip():
            st.error("Enter a zone name.")
            return
        taken = {zone.zone_id for zone in st.session_state.editor_zones}
        zone = Zone(
            zone_id=unique_id(name.strip(), taken),
            name=name.strip(),
            polygon=tuple(Point(x_pos, y_pos) for x_pos, y_pos in points),
            required_ppe=tuple(ppe),
            risk_level=int(risk),
        )
        st.session_state.editor_zones.append(zone)
        st.session_state.draft_points = []
        st.rerun()


def _save_zone_form() -> None:
    """List the working zones and write them to disk."""
    zones: list[Zone] = list(st.session_state.editor_zones)
    if zones:
        st.dataframe(_zone_table(zones), hide_index=True, use_container_width=True)
        remove = st.selectbox("Remove zone", [zone.zone_id for zone in zones])
        if st.button("Remove selected zone"):
            st.session_state.editor_zones = [zone for zone in zones if zone.zone_id != remove]
            st.rerun()
    else:
        st.write("No zones in the working list.")
    save, reload = st.columns(2)
    if save.button("Save zones", type="primary"):
        save_zones_file(ZONES, zones, already_backed_up=False)
        st.success(f"Saved {len(zones)} zones to config/zones.yaml. Previous file: config/zones.yaml.bak.")
    if reload.button("Reload from disk"):
        st.session_state.editor_zones = load_zones(ZONES) if ZONES.is_file() else []
        st.session_state.draft_points = []
        st.rerun()


def _page_alerts() -> None:
    """Filter the SQLite alert log and offer a CSV download."""
    st.subheader("Alerts history")
    manager = _open_alerts()
    try:
        available = manager.query_alerts(limit=1000)
        labels = sorted({alert.zone_name or alert.zone_id or "" for alert in available if alert.zone_name or alert.zone_id})
        zone = st.selectbox("Zone", ["All", *labels])
        start, end = _time_bounds(available)
        rows = manager.query_alerts(
            zone=None if zone == "All" else zone,
            start=start,
            end=end,
            limit=1000,
        )
    finally:
        manager.close()
    frame = _alert_table(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        "Export CSV",
        data=frame.to_csv(index=False),
        file_name="alerts.csv",
        mime="text/csv",
        disabled=frame.empty,
    )
    _render_alert_cards(rows[:12])


def _time_bounds(alerts: list[Alert]) -> tuple[float | None, float | None]:
    """Return an optional inclusive timestamp range from the loaded rows."""
    if len(alerts) < 2:
        return None, None
    stamps = [alert.timestamp for alert in alerts]
    low, high = min(stamps), max(stamps)
    if high <= low:
        return None, None
    start, end = st.slider("Timestamp", min_value=float(low), max_value=float(high), value=(float(low), float(high)))
    return float(start), float(end)


def _recent_alerts(limit: int) -> list[Alert]:
    """Read the newest critical alerts from SQLite."""
    pipeline = st.session_state.get("pipeline")
    if pipeline is not None:
        return pipeline.violations.get_recent_alerts(limit)
    manager = _open_alerts()
    try:
        return manager.get_recent_alerts(limit)
    finally:
        manager.close()


def _open_alerts() -> ViolationManager:
    """Open the alerts database configured in settings."""
    return ViolationManager(load_settings(SETTINGS), project_root=ROOT)


def _remember_result(result: FrameResult) -> None:
    """Keep the latest overlay, KPIs, and workers for the next rerun."""
    if result.annotated_frame is not None:
        st.session_state.last_rgb = _bgr_to_rgb(result.annotated_frame)
    st.session_state.last_kpis = result.kpis
    st.session_state.last_workers = list(result.workers)


def _kpi_row(kpis: dict) -> None:
    """Show the pipeline KPI fields."""
    workers, critical = st.columns(2)
    workers.metric("Workers in zones", int(kpis.get("workers_in_zones", 0)))
    critical.metric("Active critical violations", int(kpis.get("active_critical_violations", 0)))
    counts = kpis.get("zone_counts") or {}
    if counts:
        st.dataframe(
            pd.DataFrame([{"Zone": name, "Workers": count} for name, count in counts.items()]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Zone-wise counts: no worker is inside a zone on this frame.")


def _ppe_text(names: tuple[str, ...] | list[str], labels: dict[str, str]) -> str:
    """Join PPE names for a table cell, using display labels."""
    shown = [str(labels.get(name, name)) for name in names]
    return ", ".join(shown) if shown else "None"


def _worker_table(workers: list[Worker]):
    """Build the per-worker compliance table."""
    labels = load_settings(SETTINGS)["classes"].get("display_names", {})
    rows = []
    for worker in workers:
        zone = worker.zone.name if worker.zone is not None else "Outside"
        required = worker.required_ppe or (worker.zone.required_ppe if worker.zone is not None else ())
        rows.append(
            {
                "Worker": worker.track_id,
                "Zone": zone,
                "Required PPE": _ppe_text(required, labels),
                "Detected PPE": _ppe_text(worker.worn_ppe, labels),
                "Missing PPE": _ppe_text(worker.missing_ppe, labels),
                "Uncertain PPE": _ppe_text(worker.uncertain_ppe, labels),
                "Status": worker.status.value,
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["Worker", "Zone", "Required PPE", "Detected PPE", "Missing PPE", "Uncertain PPE", "Status"],
    )
    if frame.empty:
        return frame
    return frame.style.map(_paint_status, subset=["Status"])


def _paint_status(value: object) -> str:
    """CSS colour for one status cell."""
    color = STATUS_COLOR.get(str(value), "#333333")
    return f"color: {color}; font-weight: 700"


def _alert_table(alerts: list[Alert]) -> pd.DataFrame:
    """Flatten alerts for the history table and the CSV export."""
    labels = load_settings(SETTINGS)["classes"].get("display_names", {})
    rows = [
        {
            "Timestamp": alert.timestamp,
            "Worker": alert.track_id,
            "Zone": alert.zone_name or alert.zone_id or "",
            "Missing PPE": _ppe_text(alert.missing_ppe, labels),
            "Evidence": alert.evidence_path or "",
        }
        for alert in alerts
    ]
    return pd.DataFrame(rows, columns=["Timestamp", "Worker", "Zone", "Missing PPE", "Evidence"])


def _render_alert_cards(alerts: list[Alert]) -> None:
    """Show evidence thumbnails under an alert list."""
    if not alerts:
        st.caption("No alerts yet.")
        return
    labels = load_settings(SETTINGS)["classes"].get("display_names", {})
    for alert in alerts:
        zone = alert.zone_name or alert.zone_id or "zone"
        missing = _ppe_text(alert.missing_ppe, labels)
        st.markdown(
            f"<span style='color:{STATUS_COLOR['CRITICAL']}; font-weight:700'>Critical</span> "
            f"worker {alert.track_id} · {zone} · missing {missing} · t={alert.timestamp:.2f}",
            unsafe_allow_html=True,
        )
        path = Path(alert.evidence_path) if alert.evidence_path else None
        if path is not None and path.is_file():
            st.image(str(path), width=320)
        elif path is not None:
            st.caption(f"Evidence file not found: {path}")


def _zone_table(zones: list[Zone]) -> pd.DataFrame:
    """Summarise the zones waiting to be saved."""
    return pd.DataFrame(
        [
            {
                "Id": zone.zone_id,
                "Name": zone.name,
                "Risk": zone.risk_level,
                "Required PPE": ", ".join(zone.required_ppe) or "—",
                "Points": len(zone.polygon),
            }
            for zone in zones
        ]
    )


def parse_point_text(text: str) -> list[tuple[float, float]]:
    """Parse normalized `x, y` lines into polygon vertices.

    Args:
        text: One pair per line. Commas or spaces may separate the numbers.

    Returns:
        Points clamped to the unit square by the caller only when in range.

    Raises:
        ValueError: A line is not a pair inside ``[0, 1]``.
    """
    points: list[tuple[float, float]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError(f"line {line_number} must be two numbers")
        try:
            x_pos, y_pos = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"line {line_number} must be two numbers") from exc
        if not 0.0 <= x_pos <= 1.0 or not 0.0 <= y_pos <= 1.0:
            raise ValueError(f"line {line_number} is outside 0–1")
        points.append((x_pos, y_pos))
    if len(points) < 3:
        raise ValueError("paste at least 3 points")
    return points


def _list_media(suffixes: set[str]) -> list[Path]:
    """List sample files with one of ``suffixes``."""
    if not SAMPLES.is_dir():
        return []
    return sorted(path for path in SAMPLES.iterdir() if path.suffix.lower() in suffixes and path.is_file())


def _store_upload(upload: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Copy an uploaded file into ``outputs/uploads`` and return the path."""
    destination_dir = ROOT / "outputs" / "uploads"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / Path(upload.name).name
    payload = upload.getvalue()
    if not destination.is_file() or destination.stat().st_size != len(payload):
        destination.write_bytes(payload)
    return destination


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a pipeline BGR frame for Streamlit."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
else:
    main()
