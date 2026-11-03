"""Sample files and uploads for the dashboard.

Paths stay inside the sample directory or the upload directory. A camera
is never opened here.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_media(directory: Path, suffixes: set[str]) -> list[str]:
    """Return file names in ``directory`` whose suffix is allowed.

    Args:
        directory: Folder to scan. A missing folder yields an empty list.
        suffixes: Lowercase suffixes, including the dot.

    Returns:
        Sorted file names, not paths.
    """
    if not directory.is_dir():
        return []
    names = [
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(names)


def store_upload(directory: Path, filename: str, payload: bytes) -> str:
    """Write an uploaded image or video under ``directory``.

    Args:
        directory: Upload folder. Created when missing.
        filename: Client file name. Only the base name is used.
        payload: File bytes.

    Returns:
        The stored file name.

    Raises:
        ValueError: The name or suffix is not usable, or the payload is empty.
    """
    if not payload:
        raise ValueError("The uploaded file is empty.")
    stored = _safe_name(filename)
    suffix = Path(stored).suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError("Upload an image or a video.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = (directory / stored).resolve()
    if destination.parent != directory.resolve():
        raise ValueError("Upload an image or a video.")
    destination.write_bytes(payload)
    return stored


def resolve_media(directory: Path, name: str, suffixes: set[str]) -> Path:
    """Resolve one library file and reject paths outside ``directory``.

    Args:
        directory: Sample or upload folder.
        name: File name chosen in the dashboard.
        suffixes: Suffixes accepted for this request.

    Returns:
        The file path.

    Raises:
        ValueError: The name leaves the folder or uses a bad suffix.
        FileNotFoundError: The file is not in the folder.
    """
    stored = _safe_name(name)
    if Path(stored).suffix.lower() not in suffixes:
        raise ValueError("That file type is not supported.")
    directory.mkdir(parents=True, exist_ok=True)
    candidate = (directory / stored).resolve()
    if candidate.parent != directory.resolve():
        raise ValueError("That file is not in the library.")
    if not candidate.is_file():
        raise FileNotFoundError(stored)
    return candidate


def media_zones_path(directory: Path, kind: str, name: str) -> Path:
    """Return the saved-zone file for one sample or upload.

    Args:
        directory: Folder that holds per-file zone JSON.
        kind: ``sample`` or ``upload``.
        name: Media file name. Only the base name is used.

    Returns:
        A path inside ``directory``. The file may not exist yet.

    Raises:
        ValueError: ``kind`` or ``name`` is not usable.
    """
    if kind not in {"sample", "upload"}:
        raise ValueError("Choose a sample or an upload.")
    stored = _safe_name(name)
    destination = (directory / kind / f"{stored}.json").resolve()
    root = directory.resolve()
    if root not in destination.parents:
        raise ValueError("That file is not in the library.")
    return destination


def read_media_zones(path: Path) -> list[dict[str, object]]:
    """Return zones saved for one media file.

    Args:
        path: JSON file from :func:`media_zones_path`.

    Returns:
        Zone mappings. An empty list when the file is missing.
    """
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    zones = payload.get("zones") if isinstance(payload, dict) else None
    if not isinstance(zones, list):
        return []
    return [item for item in zones if isinstance(item, dict)]


def write_media_zones(path: Path, zones: list[dict[str, object]]) -> None:
    """Store the zones drawn on one media file.

    Args:
        path: Destination JSON file.
        zones: Zone mappings the page can draw again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"zones": zones}, indent=2), encoding="utf-8")


def reference_frame(path: Path, frame_index: int) -> np.ndarray:
    """Load one BGR frame from an image or a video.

    Args:
        path: Media file.
        frame_index: Zero-based video frame. Images only accept ``0``.

    Returns:
        The frame as ``(H, W, 3)``.

    Raises:
        ValueError: The index is unusable or the frame cannot be read.
    """
    if frame_index < 0:
        raise ValueError("Frame number must be 0 or greater.")
    if path.suffix.lower() in IMAGE_SUFFIXES:
        if frame_index != 0:
            raise ValueError("An image only has frame 0.")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not read {path.name}.")
        return frame
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path.name}.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_index} from {path.name}.")
    return frame


def _safe_name(filename: str) -> str:
    """Keep the base name and drop characters that change the path."""
    base = Path(filename).name.strip()
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in base)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("The file name is not usable.")
    return cleaned
