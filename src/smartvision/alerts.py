"""Amber warnings, persistence timers, and critical alerts.

Outside a zone, missing PPE is an amber warning and no timer runs. Inside
a zone, missing required PPE starts a hold timer
(``compliance.violation_hold_seconds``). The timer resets when the worker
leaves the zone or becomes compliant. Once the hold elapses, one critical
alert is stored, an evidence JPEG is written, and that worker is cooled
down for ``alerts.cooldown_seconds``. The clock is injectable so tests can
move time without sleeping. This module accepts a numpy frame and does not
open a camera.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from smartvision.zones import Zone, draw_zones


@dataclass(frozen=True)
class Alert:
    """One critical PPE alert.

    Attributes:
        track_id: Worker the timer belonged to.
        zone_id: Danger zone that contained the worker.
        missing_ppe: Required items still missing when the timer elapsed.
        evidence_path: JPEG written for this alert.
        timestamp: Clock value when the alert was raised.
        zone_name: Human-readable zone label, when known.
    """

    track_id: int
    zone_id: str | None
    missing_ppe: tuple[str, ...]
    evidence_path: str | None
    timestamp: float
    zone_name: str | None = None


@dataclass(frozen=True)
class Observation:
    """One tracked worker on the current frame.

    Attributes:
        track_id: Stable worker id.
        bbox_xyxy: Person box as ``(x1, y1, x2, y2)`` in pixels.
        zone: Zone containing the anchor, or ``None`` when outside.
        missing_ppe: Required classes not worn.
        is_compliant: ``True`` when nothing required is missing.
    """

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    zone: Zone | None
    missing_ppe: tuple[str, ...]
    is_compliant: bool


@dataclass
class _TimerState:
    """Per-worker timer and cooldown."""

    started_at: float | None = None
    cooldown_until: float | None = None


class ViolationManager:
    """Turn compliance observations into warnings and critical alerts."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        clock: Callable[[], float] | None = None,
        project_root: Path | None = None,
        draw_zones: bool = True,
    ) -> None:
        """Open the alerts database and remember the evidence directory.

        Args:
            settings: Parsed ``config/settings.yaml``. ``alerts`` and
                ``compliance.violation_hold_seconds`` are read.
            clock: Zero-argument clock. Defaults to ``time.monotonic``.
                Tests pass a fake. An explicit ``now`` on :meth:`update`
                overrides it for that call.
            project_root: Base for relative evidence and database paths.
            draw_zones: When false, evidence photos omit the zone polygon.
                Photos and recorded video pass false. The live camera leaves
                this true.
        """
        alerts = settings["alerts"]
        root = project_root or Path.cwd()
        self._hold = float(settings["compliance"]["violation_hold_seconds"])
        self._cooldown = float(alerts["cooldown_seconds"])
        self._repeat: float | None = None
        self._single = False
        self._clock = clock or time.monotonic
        self._evidence_dir = _resolve(root, str(alerts["evidence_dir"]))
        self._database = _resolve(root, str(alerts["database"]))
        self._box_color = _bgr(alerts["box_color"])
        self._zone_color = _bgr(alerts["zone_color"])
        self._draw_zones = draw_zones
        self._states: dict[int, _TimerState] = {}
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._database.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit reruns the script on a new thread, so the connection
        # opened at Start must stay usable when the next frame is processed.
        self._conn = sqlite3.connect(self._database, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                track_id INTEGER NOT NULL,
                zone_id TEXT,
                zone_name TEXT,
                missing_ppe TEXT NOT NULL,
                evidence_path TEXT
            )
            """
        )
        self._conn.commit()

    def update(
        self,
        frame: np.ndarray,
        observations: Sequence[Observation],
        now: float | None = None,
        *,
        immediate: bool = False,
    ) -> tuple[dict[int, str], list[Alert]]:
        """Advance timers and return a status per worker plus new alerts.

        Args:
            frame: BGR image saved as evidence when a critical alert fires.
            observations: Workers visible on this frame.
            now: Clock reading for this frame. When omitted, ``clock`` is
                called.
            immediate: When ``True``, an inside-zone violation alerts on
                this frame instead of waiting out the hold. Single-image
                checks use this so there is no timer.

        Returns:
            A map of track id to ``OK``, ``WARNING``, or ``CRITICAL``, and
            the critical alerts raised on this call. Amber warnings are a
            status only; they are not written to SQLite.
        """
        moment = self._clock() if now is None else float(now)
        statuses: dict[int, str] = {}
        alerts: list[Alert] = []
        for observation in observations:
            state = self._states.setdefault(observation.track_id, _TimerState())
            status, alert = self._advance(frame, observation, state, moment, immediate)
            statuses[observation.track_id] = status
            if alert is not None:
                alerts.append(alert)
        return statuses, alerts

    def get_recent_alerts(self, limit: int = 20) -> list[Alert]:
        """Return the newest critical alerts.

        Args:
            limit: Maximum rows. The newest alert is first.

        Returns:
            Stored alerts. The list is shorter when fewer rows exist.
        """
        rows = self._conn.execute(
            """
            SELECT created_at, track_id, zone_id, zone_name, missing_ppe, evidence_path
            FROM alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [_alert_from_row(row) for row in rows]

    def count_by_zone(self) -> dict[str, int]:
        """Count stored critical alerts by zone id.

        Returns:
            Map of zone id to count. Zones with no alerts are omitted.
        """
        rows = self._conn.execute(
            """
            SELECT zone_id, COUNT(*)
            FROM alerts
            GROUP BY zone_id
            """
        ).fetchall()
        return {str(zone_id): int(count) for zone_id, count in rows if zone_id is not None}

    def set_single_open_alert(self, enabled: bool = True) -> None:
        """Keep one stored alert for each continuous violation.

        The first critical alert still waits for the hold timer. Further
        frames of the same violation do not insert another row. Leaving
        the zone or becoming compliant closes that spell, so a later
        violation inserts a new row.

        Args:
            enabled: When ``True``, repeated frames update nothing in
                SQLite until the worker is clear and violates again.
        """
        self._single = bool(enabled)

    def set_repeat_seconds(self, seconds: float) -> None:
        """Repeat a critical alert while the same violation continues.

        The first alert still waits for the hold timer. Each later alert
        is raised ``seconds`` after the previous one, until the worker
        leaves the zone, becomes compliant, or the caller stops feeding
        frames.

        Args:
            seconds: Gap between repeated alerts.

        Raises:
            ValueError: ``seconds`` is not positive.
        """
        if seconds <= 0:
            raise ValueError("repeat seconds must be > 0")
        self._repeat = float(seconds)

    def set_hold_seconds(self, seconds: float) -> None:
        """Change the persistence timer used on later frames.

        Args:
            seconds: Hold time before a critical alert.

        Raises:
            ValueError: ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError("hold seconds must be >= 0")
        self._hold = float(seconds)

    def query_alerts(
        self,
        *,
        zone: str | None = None,
        start: float | None = None,
        end: float | None = None,
        limit: int = 500,
    ) -> list[Alert]:
        """Return critical alerts filtered by zone and timestamp.

        Args:
            zone: Zone id or zone name. ``None`` keeps every zone.
            start: Inclusive lower bound on ``created_at``.
            end: Inclusive upper bound on ``created_at``.
            limit: Maximum rows. The newest alert is first.

        Returns:
            Matching alerts.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if zone:
            clauses.append("(zone_id = ? OR zone_name = ?)")
            params.extend([zone, zone])
        if start is not None:
            clauses.append("created_at >= ?")
            params.append(float(start))
        if end is not None:
            clauses.append("created_at <= ?")
            params.append(float(end))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT created_at, track_id, zone_id, zone_name, missing_ppe, evidence_path
            FROM alerts
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_alert_from_row(row) for row in rows]

    def close(self) -> None:
        """Close the SQLite connection. A second call does nothing."""
        conn = self._conn
        self._conn = None
        if conn is None:
            return
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            return

    def _advance(
        self,
        frame: np.ndarray,
        observation: Observation,
        state: _TimerState,
        now: float,
        immediate: bool,
    ) -> tuple[str, Alert | None]:
        """Apply one observation and maybe raise a critical alert."""
        inside = observation.zone is not None
        if observation.is_compliant or not inside:
            state.started_at = None
            if self._single:
                state.cooldown_until = None
            if observation.is_compliant:
                return "OK", None
            return "WARNING", None
        if self._single and state.cooldown_until is not None and now < state.cooldown_until:
            return "CRITICAL", None
        if self._repeat is not None and state.cooldown_until is not None:
            if now < state.cooldown_until:
                return "CRITICAL", None
            alert = self._raise(frame, observation, now)
            state.cooldown_until = now + self._repeat
            state.started_at = None
            return "CRITICAL", alert
        cooling = state.cooldown_until is not None and now < state.cooldown_until
        if immediate:
            state.started_at = None
            if cooling:
                return "CRITICAL", None
            alert = self._raise(frame, observation, now)
            self._arm_cooldown(state, now)
            return "CRITICAL", alert
        if cooling:
            state.started_at = None
            return "CRITICAL", None
        if state.started_at is None:
            state.started_at = now
        if now - state.started_at < self._hold:
            return "WARNING", None
        alert = self._raise(frame, observation, now)
        self._arm_cooldown(state, now)
        return "CRITICAL", alert

    def _arm_cooldown(self, state: _TimerState, now: float) -> None:
        """Start the gap before another stored alert for this worker."""
        if self._single:
            state.cooldown_until = float("inf")
        elif self._repeat is not None:
            state.cooldown_until = now + self._repeat
        else:
            state.cooldown_until = now + self._cooldown
        state.started_at = None

    def _raise(self, frame: np.ndarray, observation: Observation, now: float) -> Alert:
        """Write the evidence JPEG and the SQLite row."""
        zone = observation.zone
        zone_id = zone.zone_id if zone is not None else None
        zone_name = zone.name if zone is not None else None
        filename = f"{now:.3f}_{observation.track_id}_{zone_id or 'zone'}.jpg"
        path = self._evidence_dir / filename
        _write_evidence(
            frame,
            observation.bbox_xyxy,
            zone if self._draw_zones else None,
            path,
            self._box_color,
            self._zone_color,
        )
        missing = tuple(observation.missing_ppe)
        self._conn.execute(
            """
            INSERT INTO alerts (created_at, track_id, zone_id, zone_name, missing_ppe, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now, observation.track_id, zone_id, zone_name, ",".join(missing), str(path)),
        )
        self._conn.commit()
        return Alert(
            track_id=observation.track_id,
            zone_id=zone_id,
            missing_ppe=missing,
            evidence_path=str(path),
            timestamp=now,
            zone_name=zone_name,
        )


def _write_evidence(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    zone: Zone | None,
    path: Path,
    box_color: tuple[int, int, int],
    zone_color: tuple[int, int, int],
) -> None:
    """Save a JPEG with the zone and the person box drawn."""
    canvas = frame.copy()
    if zone is not None:
        canvas = draw_zones(canvas, [zone], zone_color, thickness=2)
    x1, y1, x2, y2 = (int(round(value)) for value in bbox_xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write evidence image {path}")


def _alert_from_row(row: tuple[Any, ...]) -> Alert:
    """Rebuild an alert from one SQLite row."""
    created_at, track_id, zone_id, zone_name, missing_ppe, evidence_path = row
    missing = tuple(part for part in str(missing_ppe).split(",") if part)
    return Alert(
        track_id=int(track_id),
        zone_id=None if zone_id is None else str(zone_id),
        missing_ppe=missing,
        evidence_path=None if evidence_path is None else str(evidence_path),
        timestamp=float(created_at),
        zone_name=None if zone_name is None else str(zone_name),
    )


def _resolve(root: Path, configured: str) -> Path:
    """Resolve a relative path against ``root``."""
    path = Path(configured)
    if not path.is_absolute():
        path = root / path
    return path


def _bgr(values: Sequence[int]) -> tuple[int, int, int]:
    """Read a BGR triple from settings."""
    return int(values[0]), int(values[1]), int(values[2])
