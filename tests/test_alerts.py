"""Timer, evidence, and SQLite tests with a fake clock."""

from __future__ import annotations

import threading

import numpy as np

from smartvision.alerts import Observation, ViolationManager
from smartvision.zones import Point, Zone


class Clock:
    """Manual clock. ``update`` reads it when ``now`` is omitted."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _zone(zone_id: str = "machinery_zone") -> Zone:
    return Zone(
        zone_id=zone_id,
        name="Machinery Zone" if zone_id == "machinery_zone" else "Welding Zone",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)),
        required_ppe=("helmet", "vest", "boots"),
        risk_level=1,
    )


def _obs(zone: Zone | None, missing: tuple[str, ...] = ("helmet",), track_id: int = 1) -> Observation:
    return Observation(
        track_id=track_id,
        bbox_xyxy=(10.0, 10.0, 40.0, 80.0),
        zone=zone,
        missing_ppe=missing,
        is_compliant=not missing,
    )


def test_timer_starts_resets_escalates_and_then_cools_down(frame: np.ndarray, settings: dict) -> None:
    clock = Clock()
    manager = ViolationManager(settings, clock=clock)
    zone = _zone()

    statuses, alerts = manager.update(frame, [_obs(zone)], now=0.0)
    assert statuses[1] == "WARNING"
    assert alerts == []
    assert manager.get_recent_alerts() == []

    statuses, alerts = manager.update(frame, [_obs(zone)], now=1.9)
    assert statuses[1] == "WARNING"
    assert alerts == []

    statuses, alerts = manager.update(frame, [_obs(zone, missing=())], now=1.0)
    assert statuses[1] == "OK"
    assert alerts == []

    manager.update(frame, [_obs(zone)], now=1.1)
    statuses, alerts = manager.update(frame, [_obs(zone)], now=3.0)
    assert statuses[1] == "WARNING"
    assert alerts == []
    statuses, alerts = manager.update(frame, [_obs(zone)], now=3.1)
    assert statuses[1] == "CRITICAL"
    assert len(alerts) == 1
    assert alerts[0].missing_ppe == ("helmet",)
    assert alerts[0].evidence_path is not None
    evidence = np.fromfile(alerts[0].evidence_path, dtype=np.uint8)
    assert evidence.size > 0
    assert frame.sum() == 0

    statuses, alerts = manager.update(frame, [_obs(zone)], now=4.0)
    assert statuses[1] == "CRITICAL"
    assert alerts == []
    assert len(manager.get_recent_alerts(limit=5)) == 1

    statuses, alerts = manager.update(frame, [_obs(zone)], now=13.1)
    assert statuses[1] == "WARNING"
    assert alerts == []
    statuses, alerts = manager.update(frame, [_obs(zone)], now=15.1)
    assert len(alerts) == 1
    assert len(manager.get_recent_alerts(limit=1)) == 1
    assert manager.get_recent_alerts(limit=1)[0].timestamp == 15.1


def test_leaving_the_zone_resets_the_timer_and_warns(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    zone = _zone()
    manager.update(frame, [_obs(zone)], now=0.0)
    statuses, alerts = manager.update(frame, [_obs(None)], now=1.0)
    assert statuses[1] == "WARNING"
    assert alerts == []
    manager.update(frame, [_obs(zone)], now=1.5)
    statuses, alerts = manager.update(frame, [_obs(zone)], now=3.4)
    assert alerts == []
    statuses, alerts = manager.update(frame, [_obs(zone)], now=3.5)
    assert len(alerts) == 1
    assert alerts[0].zone_id == "machinery_zone"


def test_outside_a_zone_never_raises_a_critical_alert(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    statuses, alerts = manager.update(frame, [_obs(None)], now=100.0)
    assert statuses[1] == "WARNING"
    assert alerts == []
    assert manager.count_by_zone() == {}


def test_count_by_zone_and_recent_limit(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    manager.update(frame, [_obs(_zone("machinery_zone"), track_id=1)], now=0.0, immediate=True)
    manager.update(frame, [_obs(_zone("welding_zone"), track_id=2)], now=0.0, immediate=True)
    manager.update(frame, [_obs(_zone("welding_zone"), track_id=3)], now=0.0, immediate=True)
    assert manager.count_by_zone() == {"machinery_zone": 1, "welding_zone": 2}
    recent = manager.get_recent_alerts(limit=2)
    assert [alert.track_id for alert in recent] == [3, 2]


def test_query_filters_by_zone_and_time(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    manager.update(frame, [_obs(_zone("machinery_zone"), track_id=1)], now=2.0, immediate=True)
    manager.update(frame, [_obs(_zone("welding_zone"), track_id=2)], now=8.0, immediate=True)
    machinery = manager.query_alerts(zone="machinery_zone")
    assert [alert.track_id for alert in machinery] == [1]
    later = manager.query_alerts(start=5.0, end=8.0)
    assert [alert.track_id for alert in later] == [2]


def test_hold_seconds_can_be_lengthened(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    zone = _zone()
    manager.update(frame, [_obs(zone)], now=0.0)
    manager.set_hold_seconds(10.0)
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=2.0)
    assert alerts == []
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=10.0)
    assert len(alerts) == 1


def test_single_open_alert_refreshes_without_a_duplicate_row(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    manager.set_single_open_alert(True)
    zone = _zone()
    manager.update(frame, [_obs(zone)], now=0.0)
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=2.0)
    assert len(alerts) == 1
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=7.0)
    assert alerts == []
    assert len(manager.get_recent_alerts()) == 1
    manager.update(frame, [_obs(zone, missing=())], now=8.0)
    manager.update(frame, [_obs(zone)], now=8.0)
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=10.0)
    assert len(alerts) == 1
    assert [alert.timestamp for alert in reversed(manager.get_recent_alerts())] == [2.0, 10.0]


def test_repeat_raises_a_new_alert_every_three_seconds(frame: np.ndarray, settings: dict) -> None:
    manager = ViolationManager(settings)
    manager.set_repeat_seconds(3.0)
    zone = _zone()
    manager.update(frame, [_obs(zone)], now=0.0)
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=2.0)
    assert len(alerts) == 1
    statuses, alerts = manager.update(frame, [_obs(zone)], now=4.9)
    assert statuses[1] == "CRITICAL"
    assert alerts == []
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=5.0)
    assert len(alerts) == 1
    _statuses, alerts = manager.update(frame, [_obs(zone)], now=8.0)
    assert len(alerts) == 1
    assert [alert.timestamp for alert in reversed(manager.get_recent_alerts(limit=3))] == [2.0, 5.0, 8.0]


def test_clock_is_used_when_now_is_omitted(frame: np.ndarray, settings: dict) -> None:
    clock = Clock()
    manager = ViolationManager(settings, clock=clock)
    zone = _zone()
    manager.update(frame, [_obs(zone)])
    clock.now = 2.0
    _statuses, alerts = manager.update(frame, [_obs(zone)])
    assert len(alerts) == 1
    assert alerts[0].timestamp == 2.0


def test_connection_can_be_read_and_closed_from_another_thread(settings: dict) -> None:
    manager = ViolationManager(settings)
    errors: list[BaseException] = []

    def _use_from_other_thread() -> None:
        try:
            assert manager.get_recent_alerts() == []
            manager.close()
            manager.close()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=_use_from_other_thread)
    worker.start()
    worker.join()
    assert errors == []
