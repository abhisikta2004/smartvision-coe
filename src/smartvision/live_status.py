"""Live safety lines for people inside danger zones.

Detection keeps running on every frame. This module only decides what the
safety panel should say, and it refreshes that text on a fixed interval.
An ongoing violation keeps one entry and a new timestamp. A violation that
ends and later returns gets a new event id. The module accepts worker
objects and does not open a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StatusLine:
    """One row in the live safety panel.

    Attributes:
        key: Stable identity for this row during a refresh.
        kind: ``safe``, ``compliant``, ``violation``, or ``uncertain``.
        message: Text shown to the operator.
        updated_at: Clock value of this refresh.
        track_id: Worker id, or ``None`` for a zone-level safe line.
        zone_name: Zone the line refers to.
        required_ppe: PPE the zone requires.
        detected_ppe: Required items assigned to the worker.
        missing_ppe: Required items confirmed absent.
        uncertain_ppe: Required items that could not be verified.
        event_id: Violation spell. ``0`` when the line is not a violation.
    """

    key: str
    kind: str
    message: str
    updated_at: float
    track_id: int | None = None
    zone_name: str = ""
    required_ppe: tuple[str, ...] = ()
    detected_ppe: tuple[str, ...] = ()
    missing_ppe: tuple[str, ...] = ()
    uncertain_ppe: tuple[str, ...] = ()
    event_id: int = 0


@dataclass
class StatusBoard:
    """The panel contents and the bookkeeping for the next refresh.

    Attributes:
        lines: Rows from the latest refresh.
        last_refresh: Clock value of that refresh, or ``None`` before the
            first one.
        next_event: Next violation spell id to issue.
        open_events: Active violation spell per worker and zone.
    """

    lines: tuple[StatusLine, ...] = ()
    last_refresh: float | None = None
    next_event: int = 1
    open_events: dict[str, int] = field(default_factory=dict)


def ppe_phrase(names: Sequence[str], labels: Mapping[str, str]) -> str:
    """Join PPE names in sentence form.

    Args:
        names: Class names, in display order.
        labels: Optional display name per class.

    Returns:
        ``Helmet``, ``Helmet and Safety Vest``, or a comma list ending
        with ``and``.
    """
    shown = [str(labels.get(name, name.replace("_", " ").title())) for name in names]
    if not shown:
        return ""
    if len(shown) == 1:
        return shown[0]
    if len(shown) == 2:
        return f"{shown[0]} and {shown[1]}"
    return ", ".join(shown[:-1]) + f", and {shown[-1]}"


def refresh_board(
    board: StatusBoard,
    workers: Sequence[object],
    zones: Sequence[object],
    now: float,
    interval: float,
    labels: Mapping[str, str] | None = None,
    wall_time: float | None = None,
) -> StatusBoard:
    """Return the safety panel, refreshing only when the interval has elapsed.

    Args:
        board: Previous panel. The first call can be an empty board.
        workers: People from the latest frame. Each needs ``track_id``,
            ``zone``, ``missing_ppe``, ``worn_ppe``, and ``uncertain_ppe``.
        zones: Zones currently being watched.
        now: Monotonic clock used to measure the interval.
        interval: Seconds between panel refreshes. Detection is not delayed
            by this value; a caller that is early receives ``board`` back.
        labels: Display names for PPE classes.
        wall_time: Timestamp stored on the new rows. Defaults to ``now``.

    Returns:
        The same board when ``interval`` has not elapsed. Otherwise a new
        board built from ``workers``.

    Raises:
        ValueError: ``interval`` is not positive.
    """
    if interval <= 0:
        raise ValueError("status interval must be > 0")
    if board.last_refresh is not None and now - board.last_refresh < interval:
        return board
    stamp = now if wall_time is None else float(wall_time)
    names = dict(labels or {})
    open_events = dict(board.open_events)
    next_event = board.next_event
    lines, open_events, next_event = _lines(workers, zones, stamp, names, open_events, next_event)
    return StatusBoard(
        lines=tuple(lines),
        last_refresh=float(now),
        next_event=next_event,
        open_events=open_events,
    )


def _lines(
    workers: Sequence[object],
    zones: Sequence[object],
    stamp: float,
    labels: Mapping[str, str],
    open_events: dict[str, int],
    next_event: int,
) -> tuple[list[StatusLine], dict[str, int], int]:
    """Build the rows for one refresh."""
    inside = [worker for worker in workers if getattr(worker, "zone", None) is not None]
    lines: list[StatusLine] = []
    if not workers:
        lines.append(
            StatusLine(
                key="frame-empty",
                kind="safe",
                message="[SAFE] No person detected in the camera view.",
                updated_at=stamp,
            )
        )
        return lines, {}, next_event
    occupied = {worker.zone.zone_id for worker in inside}
    if not inside:
        for zone in zones:
            lines.append(_safe_zone(zone.name, stamp))
        if not zones:
            lines.append(
                StatusLine(
                    key="frame-clear",
                    kind="safe",
                    message="[SAFE] No person detected inside a danger zone.",
                    updated_at=stamp,
                )
            )
        return lines, {}, next_event
    for zone in zones:
        if zone.zone_id not in occupied:
            lines.append(_safe_zone(zone.name, stamp))
    still_open: dict[str, int] = {}
    for worker in sorted(inside, key=lambda item: int(item.track_id)):
        line, open_events, next_event = _worker_line(worker, stamp, labels, open_events, next_event)
        if line.kind == "violation":
            still_open[f"{worker.track_id}:{worker.zone.zone_id}"] = line.event_id
        lines.append(line)
    return lines, still_open, next_event


def _safe_zone(zone_name: str, stamp: float) -> StatusLine:
    """One safe line for a zone that contains nobody."""
    return StatusLine(
        key=f"safe:{zone_name}",
        kind="safe",
        zone_name=zone_name,
        message=f"[SAFE] No person detected inside {zone_name}.",
        updated_at=stamp,
    )


def _checklist(
    worker: object,
    zone: object,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Keep required, detected, missing, and uncertain inside the zone checklist.

    Args:
        worker: Person with ``worn_ppe``, ``missing_ppe``, and ``uncertain_ppe``.
        zone: Zone whose ``required_ppe`` is the only set that may be reported.

    Returns:
        Required, detected, missing, and uncertain class names. A class that
        is not on the checklist is dropped from every list.
    """
    required = tuple(dict.fromkeys(str(name) for name in getattr(zone, "required_ppe", ())))
    worn = {str(name) for name in getattr(worker, "worn_ppe", ())}
    missing_in = {str(name) for name in getattr(worker, "missing_ppe", ())}
    uncertain_in = {str(name) for name in getattr(worker, "uncertain_ppe", ())}
    detected = tuple(name for name in required if name in worn)
    missing = tuple(name for name in required if name in missing_in and name not in worn)
    uncertain = tuple(
        name
        for name in required
        if name in uncertain_in and name not in worn and name not in set(missing)
    )
    return required, detected, missing, uncertain


def _worker_line(
    worker: object,
    stamp: float,
    labels: Mapping[str, str],
    open_events: dict[str, int],
    next_event: int,
) -> tuple[StatusLine, dict[str, int], int]:
    """Describe one worker who overlaps a zone."""
    zone = worker.zone
    track_id = int(worker.track_id)
    required, detected, missing, uncertain = _checklist(worker, zone)
    required_text = ppe_phrase(required, labels)
    spell = f"{track_id}:{zone.zone_id}"
    if missing:
        event_id = open_events.get(spell)
        entered = event_id is None
        if event_id is None:
            event_id = next_event
            next_event += 1
        message = _violation_message(
            track_id,
            zone.name,
            required_text,
            ppe_phrase(missing, labels),
            ppe_phrase(detected, labels),
            ppe_phrase(uncertain, labels),
            entered=entered,
            single_missing=len(missing) == 1,
        )
        kind = "violation"
    elif uncertain:
        event_id = 0
        message = (
            f"[UNCERTAIN] Person {track_id} is inside {zone.name}. "
            f"Required PPE: {required_text}. "
            f"Could not verify: {ppe_phrase(uncertain, labels)}."
        )
        kind = "uncertain"
    else:
        event_id = 0
        wearing = required_text or "the required PPE"
        message = (
            f"[COMPLIANT] Person {track_id} is inside {zone.name} and is wearing "
            f"all required PPE: {wearing}."
        )
        kind = "compliant"
    line = StatusLine(
        key=f"{kind}:{spell}",
        kind=kind,
        message=message,
        updated_at=stamp,
        track_id=track_id,
        zone_name=zone.name,
        required_ppe=required,
        detected_ppe=detected,
        missing_ppe=missing,
        uncertain_ppe=uncertain,
        event_id=event_id,
    )
    return line, open_events, next_event


def _violation_message(
    track_id: int,
    zone_name: str,
    required_text: str,
    missing_text: str,
    detected_text: str,
    uncertain_text: str,
    *,
    entered: bool,
    single_missing: bool,
) -> str:
    """Build the violation sentence for one worker."""
    if single_missing and entered:
        where = f"Person {track_id} has entered {zone_name}"
    else:
        where = f"Person {track_id} is inside {zone_name}"
    message = (
        f"[SAFETY VIOLATION] {where}. "
        f"Required PPE: {required_text}. Missing PPE: {missing_text}."
    )
    if single_missing:
        message += f" The worker must wear the required {missing_text.lower()} while inside this zone."
    if detected_text:
        message += f" Detected PPE: {detected_text}."
    if uncertain_text:
        message += f" Could not verify: {uncertain_text}."
    return message
