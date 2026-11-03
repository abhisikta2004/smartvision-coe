"""Tests for anchors, polygon membership, and the zone editor files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smartvision.zones import (
    Point,
    Zone,
    ZoneChecker,
    bottom_center,
    draw_zones,
    box_overlaps_zone,
    find_zone,
    find_zone_for_box,
    load_zones,
    person_anchor,
    save_zones,
)

ROOT = Path(__file__).resolve().parents[1]


def _square(zone_id: str, risk: int, x0: float, y0: float, x1: float, y1: float) -> Zone:
    return Zone(
        zone_id=zone_id,
        name=zone_id,
        polygon=(Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)),
        required_ppe=("helmet",),
        risk_level=risk,
    )


def test_person_anchor_is_the_bottom_center() -> None:
    anchor = person_anchor((0.0, 0.0, 10.0, 20.0))
    assert anchor == Point(5.0, 20.0)
    assert bottom_center((0.0, 0.0, 10.0, 20.0)) == anchor


def test_find_zone_picks_the_highest_risk_on_overlap() -> None:
    low = _square("low", 1, 0.0, 0.0, 0.6, 0.6)
    high = _square("high", 5, 0.4, 0.4, 1.0, 1.0)
    winner = find_zone(Point(50, 50), [low, high], (100, 100))
    assert winner is not None
    assert winner.zone_id == "high"
    assert find_zone(Point(5, 5), [low, high], (100, 100)) == low
    assert find_zone(Point(95, 5), [low, high], (100, 100)) is None


def test_partial_box_overlap_counts_as_inside_the_zone() -> None:
    zone = _square("corner", 1, 0.0, 0.0, 0.1, 0.1)
    assert box_overlaps_zone((8.0, 0.0, 32.0, 40.0), zone, (100, 100))
    assert not box_overlaps_zone((20.0, 20.0, 40.0, 40.0), zone, (100, 100))
    winner = find_zone_for_box((8.0, 0.0, 32.0, 40.0), [zone], (100, 100))
    assert winner is not None
    assert winner.zone_id == "corner"
    assert find_zone(Point(20.0, 40.0), [zone], (100, 100)) is None


def test_box_overlap_keeps_the_higher_risk_zone() -> None:
    low = _square("low", 1, 0.0, 0.0, 1.0, 1.0)
    high = _square("high", 4, 0.0, 0.0, 0.2, 0.2)
    winner = find_zone_for_box((0.0, 0.0, 10.0, 10.0), [low, high], (100, 100))
    assert winner is not None
    assert winner.zone_id == "high"


def test_equal_risk_keeps_the_earlier_zone() -> None:
    first = _square("first", 2, 0.0, 0.0, 1.0, 1.0)
    second = _square("second", 2, 0.0, 0.0, 1.0, 1.0)
    winner = ZoneChecker([first, second]).find_zone(Point(0.5, 0.5), (1, 1))
    assert winner is not None
    assert winner.zone_id == "first"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = [
        Zone(
            zone_id="welding_zone",
            name="Welding Zone",
            polygon=(Point(0.1, 0.2), Point(0.3, 0.2), Point(0.2, 0.4)),
            required_ppe=("helmet", "goggles"),
            risk_level=2,
        )
    ]
    path = tmp_path / "zones.yaml"
    save_zones(path, original)
    loaded = load_zones(path)
    assert loaded[0].zone_id == "welding_zone"
    assert loaded[0].name == "Welding Zone"
    assert loaded[0].required_ppe == ("helmet", "goggles")
    assert loaded[0].risk_level == 2
    assert loaded[0].polygon[0].x == 0.1
    assert loaded[0].polygon[2].y == 0.4


def test_draw_zones_does_not_change_the_input_frame() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    zone = _square("pad", 1, 0.0, 0.0, 1.0, 1.0)
    drawn = draw_zones(frame, [zone], (255, 128, 0), thickness=2)
    assert frame.sum() == 0
    assert drawn.sum() > 0
    assert drawn.shape == frame.shape


def test_photo_and_video_zones_do_not_include_the_sample_zones() -> None:
    zones = load_zones(ROOT / "config" / "zones.yaml")
    names = {zone.name for zone in zones}
    assert "Machinery Zone" not in names
    assert "Welding Zone" not in names
    for zone in zones:
        assert "gloves" not in zone.required_ppe
        assert len(zone.polygon) >= 3
