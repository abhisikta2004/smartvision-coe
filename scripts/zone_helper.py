"""Click danger-zone polygons on a video frame or a still image.

Left click adds a point. ``n`` finishes the polygon and asks for the zone
name, risk level, and required PPE in the terminal. ``u`` removes the last
point. ``s`` writes ``config/zones.yaml`` and keeps the previous file as
``config/zones.yaml.bak``. ``q`` closes the window without writing.

Existing zones are drawn while you edit. New polygons are stored normalized,
in the same YAML shape ``smartvision.zones.load_zones`` already reads.

Example:
    python scripts/zone_helper.py data/samples/site.mp4
    python scripts/zone_helper.py data/samples/site.mp4 --frame 40
    python scripts/zone_helper.py data/samples/frame.jpg
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smartvision.zones import Point, Zone, draw_zones, load_zones  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW = "SmartVision zones"
DRAFT_COLOR = (0, 255, 0)
HEADER = """\
# Danger zones. Points are normalized image coordinates: x then y, each from
# 0 (left/top) to 1 (right/bottom). Edit the numbers in place.
# risk_level: the higher number wins when a person stands in two zones.
# required_ppe is checked only inside that zone.
# Outside every zone, compliance.default_required_ppe applies (helmet and vest).
# Gloves are detected by the model but are not required in these examples.
"""


def grab_frame(source: Path, frame_index: int) -> np.ndarray:
    """Load one BGR frame from an image file or a video.

    Args:
        source: Image or video path.
        frame_index: Zero-based video frame. Images only accept ``0``.

    Returns:
        The frame as a ``(H, W, 3)`` array.

    Raises:
        FileNotFoundError: ``source`` does not exist.
        ValueError: The index is negative, the image is asked for a later
            frame, or the frame cannot be decoded.
    """
    if frame_index < 0:
        raise ValueError("--frame must be >= 0")
    if not source.is_file():
        raise FileNotFoundError(f"media file not found: {source}")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        if frame_index != 0:
            raise ValueError("an image only has frame 0")
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"could not read image: {source}")
        return frame
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    try:
        frame = _read_video_frame(capture, frame_index)
        if frame is None:
            raise ValueError(f"could not read frame {frame_index} from {source}")
        return frame
    finally:
        capture.release()


def load_editor_config(settings_path: Path) -> tuple[list[str], tuple[int, int, int]]:
    """Read the canonical PPE names and the zone overlay color.

    Args:
        settings_path: ``config/settings.yaml``.

    Returns:
        PPE class names, then a BGR color for zones already on disk.
    """
    payload = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    names = [str(name) for name in payload["classes"]["ppe"]]
    raw = payload["visualize"]["colors"]["zone"]
    color = (int(raw[0]), int(raw[1]), int(raw[2]))
    return names, color


def slug(name: str) -> str:
    """Turn a zone name into the id style used in ``config/zones.yaml``.

    Args:
        name: Zone name typed in the terminal.

    Returns:
        A lowercase id. Non-letters become single underscores.
    """
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "zone"


def unique_id(name: str, taken: set[str]) -> str:
    """Return a zone id that is not already in ``taken``.

    Args:
        name: Zone name.
        taken: Ids already loaded or added in this session.

    Returns:
        ``slug(name)``, or that slug with a numeric suffix.
    """
    base = slug(name)
    if base not in taken:
        return base
    number = 2
    while f"{base}_{number}" in taken:
        number += 1
    return f"{base}_{number}"


def parse_ppe_choice(text: str, canonical: list[str]) -> tuple[str, ...]:
    """Parse a terminal PPE selection.

    Args:
        text: Comma-separated numbers (1-based) or canonical names.
            Empty text selects nothing.
        canonical: PPE class names from settings, in display order.

    Returns:
        Selected names in canonical order, without duplicates.

    Raises:
        ValueError: A token is not a listed number or name.
    """
    pieces = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not pieces:
        return ()
    by_name = {name.lower(): name for name in canonical}
    selected: list[str] = []
    for piece in pieces:
        if piece.isdigit():
            index = int(piece)
            if index < 1 or index > len(canonical):
                raise ValueError(f"{piece} is not in 1..{len(canonical)}")
            name = canonical[index - 1]
        elif piece in by_name:
            name = by_name[piece]
        else:
            raise ValueError(f"{piece!r} is not in the canonical PPE list")
        if name not in selected:
            selected.append(name)
    order = {name: index for index, name in enumerate(canonical)}
    selected.sort(key=lambda name: order[name])
    return tuple(selected)


def normalize_points(
    points: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[Point, ...]:
    """Convert clicked pixels to normalized zone vertices.

    Args:
        points: Pixel coordinates on the original frame.
        width: Frame width.
        height: Frame height.

    Returns:
        Vertices clamped to ``[0, 1]``.
    """
    vertices: list[Point] = []
    for x_pos, y_pos in points:
        x_norm = min(1.0, max(0.0, x_pos / width))
        y_norm = min(1.0, max(0.0, y_pos / height))
        vertices.append(Point(x_norm, y_norm))
    return tuple(vertices)


def format_zones_yaml(zones: list[Zone]) -> str:
    """Serialize zones in the ``config/zones.yaml`` layout.

    Args:
        zones: Existing zones plus any finished in this session.

    Returns:
        YAML text ``load_zones`` accepts, with normalized ``[x, y]`` points.
    """
    if not zones:
        return HEADER + "zones: []\n"
    lines = [HEADER.rstrip("\n"), "zones:"]
    for zone in zones:
        lines.append(f"  - id: {_yaml_scalar(zone.zone_id)}")
        lines.append(f"    name: {_yaml_scalar(zone.name)}")
        lines.append(f"    risk_level: {int(zone.risk_level)}")
        if zone.required_ppe:
            lines.append("    required_ppe:")
            for name in zone.required_ppe:
                lines.append(f"      - {_yaml_scalar(name)}")
        else:
            lines.append("    required_ppe: []")
        lines.append("    polygon:")
        for point in zone.polygon:
            lines.append(f"    - [{point.x:.4f}, {point.y:.4f}]")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_zones_file(path: Path, zones: list[Zone], *, already_backed_up: bool) -> bool:
    """Back up ``path`` once, then overwrite it.

    Args:
        path: Destination, normally ``config/zones.yaml``.
        zones: Full zone list to write.
        already_backed_up: When ``True``, leave the existing ``.bak`` alone
            so a second save in the same session does not replace the
            pre-session file.

    Returns:
        ``True`` when this call created the backup.
    """
    backed_up = already_backed_up
    if path.is_file() and not already_backed_up:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"backed up {path.name} -> {backup.name}")
        backed_up = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_zones_yaml(zones), encoding="utf-8")
    print(f"wrote {path} ({len(zones)} zones)")
    return backed_up


def prompt_zone(
    points: list[tuple[int, int]],
    frame_shape: tuple[int, int],
    canonical: list[str],
    taken_ids: set[str],
) -> Zone | None:
    """Ask for a name, risk level, and required PPE.

    Args:
        points: At least three clicked pixels.
        frame_shape: ``(height, width)`` of the source frame.
        canonical: PPE names offered in the terminal.
        taken_ids: Ids that must not be reused. The new id is added.

    Returns:
        The new zone, or ``None`` when the name is left blank.
    """
    print("\nFinish zone. Leave the name blank to cancel and keep these points.")
    name = input("Zone name: ").strip()
    if not name:
        print("cancelled")
        return None
    while True:
        raw_risk = input("Risk level (higher number wins on overlap): ").strip()
        try:
            risk = int(raw_risk)
            break
        except ValueError:
            print("enter an integer")
    required = _prompt_ppe(canonical)
    height, width = frame_shape
    zone_id = unique_id(name, taken_ids)
    taken_ids.add(zone_id)
    zone = Zone(
        zone_id=zone_id,
        name=name,
        polygon=normalize_points(points, width, height),
        required_ppe=required,
        risk_level=risk,
    )
    required_text = ", ".join(zone.required_ppe) if zone.required_ppe else "(none)"
    print(f"added {zone.name} id={zone.zone_id} risk={zone.risk_level} ppe={required_text}")
    return zone


class ZoneEditor:
    """OpenCV polygon editor over one frozen frame."""

    def __init__(
        self,
        frame: np.ndarray,
        zones: list[Zone],
        canonical: list[str],
        zone_color: tuple[int, int, int],
        destination: Path,
    ) -> None:
        """Store the frame and the zones already on disk.

        Args:
            frame: BGR image to click on.
            zones: Zones loaded from ``config/zones.yaml``.
            canonical: PPE class names.
            zone_color: BGR color for saved zones.
            destination: YAML file written when ``s`` is pressed.
        """
        self.frame = frame
        self.zones = list(zones)
        self.canonical = canonical
        self.zone_color = zone_color
        self.destination = destination
        self.points: list[tuple[int, int]] = []
        self.busy = False
        self._backed_up = False
        height, width = frame.shape[:2]
        self._width = width
        self._height = height
        self._scale = min(1.0, 1600 / width, 900 / height)

    def run(self) -> None:
        """Show the window until the user quits.

        ``q`` and Esc leave the file untouched. ``s`` writes it.
        """
        print("Left click adds a point. n finishes the zone, u undoes a point, s saves, q quits.")
        print(f"{len(self.zones)} existing zones shown. Terminal prompts appear when you press n.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self._render())
                key = cv2.waitKey(30) & 0xFF
                if key == 255:
                    if not _window_open():
                        print("window closed; nothing saved")
                        break
                    continue
                if key in (ord("q"), 27):
                    print("quit without saving")
                    break
                if key in (ord("u"), ord("U")):
                    self._undo()
                elif key in (ord("n"), ord("N")):
                    self._finish()
                elif key in (ord("s"), ord("S")):
                    self._save()
        finally:
            cv2.destroyAllWindows()

    def _on_mouse(self, event: int, x_pos: int, y_pos: int, _flags: int, _param: object) -> None:
        """Append one vertex on the left button."""
        if self.busy or event != cv2.EVENT_LBUTTONDOWN:
            return
        original_x = int(round(x_pos / self._scale))
        original_y = int(round(y_pos / self._scale))
        original_x = min(self._width - 1, max(0, original_x))
        original_y = min(self._height - 1, max(0, original_y))
        self.points.append((original_x, original_y))

    def _undo(self) -> None:
        """Drop the latest clicked point."""
        if not self.points:
            print("no point to undo")
            return
        removed = self.points.pop()
        print(f"undo {removed}")

    def _finish(self) -> None:
        """Prompt for zone details when the polygon has three points."""
        if len(self.points) < 3:
            print("need at least 3 points before finishing a zone")
            return
        self.busy = True
        try:
            zone = prompt_zone(
                self.points,
                (self._height, self._width),
                self.canonical,
                {item.zone_id for item in self.zones},
            )
        finally:
            self.busy = False
        if zone is None:
            return
        self.zones.append(zone)
        self.points = []

    def _save(self) -> None:
        """Write every finished zone. An open polygon is not included."""
        if self.points:
            print(f"{len(self.points)} unfinished points were not saved; press n to finish that zone")
        self._backed_up = save_zones_file(
            self.destination,
            self.zones,
            already_backed_up=self._backed_up,
        )

    def _render(self) -> np.ndarray:
        """Draw saved zones, the draft polygon, and the key hints."""
        canvas = draw_zones(self.frame, self.zones, self.zone_color, thickness=2)
        _draw_banner(
            canvas,
            f"click add | n finish | u undo | s save | q quit | points {len(self.points)} | zones {len(self.zones)}",
        )
        for index, point in enumerate(self.points):
            if index:
                cv2.line(canvas, self.points[index - 1], point, DRAFT_COLOR, 2)
            cv2.circle(canvas, point, 4, DRAFT_COLOR, -1)
        if len(self.points) >= 3:
            cv2.line(canvas, self.points[-1], self.points[0], DRAFT_COLOR, 1)
        if self._scale < 1:
            shown = (
                max(1, int(round(self._width * self._scale))),
                max(1, int(round(self._height * self._scale))),
            )
            canvas = cv2.resize(canvas, shown, interpolation=cv2.INTER_AREA)
        return canvas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the media path and the optional frame index.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Namespace with ``source`` and ``frame``.
    """
    parser = argparse.ArgumentParser(description="Draw danger zones on a video frame or an image.")
    parser.add_argument("source", help="video or image path")
    parser.add_argument("--frame", type=int, default=0, help="zero-based video frame (default: first frame)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Open the editor on the requested frame."""
    args = parse_args(argv)
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    frame = grab_frame(source, args.frame)
    canonical, zone_color = load_editor_config(ROOT / "config" / "settings.yaml")
    destination = ROOT / "config" / "zones.yaml"
    existing = load_zones(destination) if destination.is_file() else []
    ZoneEditor(frame, existing, canonical, zone_color, destination).run()


def _prompt_ppe(canonical: list[str]) -> tuple[str, ...]:
    """Ask until the PPE entry is empty or a valid subset."""
    print("Required PPE. Enter numbers or names, separated by commas. Empty means none.")
    for index, name in enumerate(canonical, start=1):
        print(f"  {index}. {name}")
    while True:
        try:
            return parse_ppe_choice(input("Required PPE: "), canonical)
        except ValueError as exc:
            print(exc)


def _yaml_scalar(value: str) -> str:
    """Return a YAML scalar, quoted when the text is not plain."""
    if value == "":
        return '""'
    special = set(":#{}[]&*!|>%@`'\\\"\n\t")
    if value.lower() in {"yes", "no", "true", "false", "null", "~"} or any(char in special for char in value):
        return yaml.safe_dump(value).strip()
    return value


def _draw_banner(canvas: np.ndarray, text: str) -> None:
    """Paint a one-line hint along the top of ``canvas``."""
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    """Return video frame ``frame_index``, reading forward when seek fails."""
    if frame_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    reached = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if frame_index and (not ok or frame is None or reached <= frame_index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = None
        for _ in range(frame_index + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
    if not ok or frame is None:
        return None
    return frame


def _window_open() -> bool:
    """Report whether the OpenCV window is still visible."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


if __name__ == "__main__":
    main()
