"""Tests for PPE assignment and required-set comparison."""

from __future__ import annotations

from pathlib import Path

from smartvision.compliance import ComplianceEngine, assign_ppe, classes_on_person, reconcile_worn
from smartvision.detector import Detection
from smartvision.pipeline import load_settings
from smartvision.zones import Zone, Point

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {
    "helmet": (0.0, 0.333333),
    "goggles": (0.0, 0.333333),
    "vest": (0.333333, 0.666667),
    "gloves": (0.333333, 1.0),
    "boots": (0.75, 1.0),
}


def _box(class_name: str, bbox: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
    return Detection(class_name, confidence, bbox)


def test_body_regions_accept_a_class_only_in_its_band() -> None:
    person = _box("person", (0.0, 0.0, 100.0, 100.0))
    helmet_ok = _box("helmet", (40.0, 5.0, 60.0, 20.0))
    helmet_low = _box("helmet", (40.0, 80.0, 60.0, 95.0))
    assigned = assign_ppe([person], [helmet_ok, helmet_low], REGIONS)
    assert [det.bbox_xyxy for det in assigned[0]] == [helmet_ok.bbox_xyxy]


def test_overlapping_people_keep_the_larger_overlap(settings: dict) -> None:
    left = _box("person", (0.0, 0.0, 100.0, 100.0))
    right = _box("person", (80.0, 0.0, 180.0, 100.0))
    helmet = _box("helmet", (60.0, 5.0, 100.0, 25.0))
    engine = ComplianceEngine(settings)
    results = engine.evaluate([left, right], [helmet], [("helmet",), ("helmet",)])
    assert results[0].worn == ("helmet",)
    assert results[0].missing == ()
    assert results[0].is_compliant
    assert results[1].missing == ("helmet",)
    assert not results[1].is_compliant


def test_outside_a_zone_uses_the_default_set_and_ignores_unrequired_gloves(settings: dict) -> None:
    engine = ComplianceEngine(settings)
    person = _box("person", (0.0, 0.0, 100.0, 300.0))
    helmet = _box("helmet", (40.0, 10.0, 60.0, 40.0))
    vest = _box("vest", (30.0, 120.0, 70.0, 180.0))
    gloves = _box("gloves", (20.0, 160.0, 40.0, 200.0))
    zone = Zone(
        zone_id="welding_zone",
        name="Welding Zone",
        polygon=(Point(0, 0), Point(1, 0), Point(1, 1)),
        required_ppe=("helmet", "vest", "goggles"),
        risk_level=2,
    )
    outside = engine.evaluate(
        [person],
        [helmet, vest, gloves],
        [engine.required_for(None)],
    )[0]
    assert outside.required_ppe == ("helmet", "vest")
    assert outside.worn == ("helmet", "vest")
    assert "gloves" not in outside.worn
    assert outside.missing == ()
    assert outside.is_compliant
    inside = engine.evaluate(
        [person],
        [helmet, vest, gloves],
        [engine.required_for(zone)],
    )[0]
    assert inside.missing == ("goggles",)
    assert not inside.is_compliant


def test_a_cropped_head_band_is_uncertain_instead_of_missing(settings: dict) -> None:
    engine = ComplianceEngine(settings)
    cropped = _box("person", (10.0, -40.0, 40.0, 80.0))
    hidden = engine.evaluate([cropped], [], [("helmet", "vest")], image_size=(100, 100))[0]
    assert hidden.uncertain == ("helmet",)
    assert hidden.missing == ("vest",)
    assert not hidden.is_compliant
    visible = _box("person", (10.0, 10.0, 40.0, 90.0))
    confirmed = engine.evaluate([visible], [], [("helmet",)], image_size=(100, 100))[0]
    assert confirmed.missing == ("helmet",)
    assert confirmed.uncertain == ()


def _gear() -> list[Detection]:
    person = _box("person", (0.0, 0.0, 100.0, 300.0))
    helmet = _box("helmet", (40.0, 10.0, 60.0, 40.0))
    vest = _box("vest", (30.0, 120.0, 70.0, 180.0))
    gloves = _box("gloves", (20.0, 160.0, 40.0, 220.0))
    return [person, helmet, vest, gloves]


def test_selected_ppe_is_the_only_set_reported(settings: dict) -> None:
    engine = ComplianceEngine(settings)
    person, helmet, vest, gloves = _gear()
    detected = [helmet, vest, gloves]

    helmet_missing_vest = engine.evaluate([person], [helmet, gloves], [("helmet", "vest")])[0]
    assert helmet_missing_vest.required_ppe == ("helmet", "vest")
    assert helmet_missing_vest.worn == ("helmet",)
    assert helmet_missing_vest.missing == ("vest",)
    assert not helmet_missing_vest.is_compliant
    assert "gloves" not in helmet_missing_vest.worn + helmet_missing_vest.missing

    complete = engine.evaluate([person], [helmet, vest], [("helmet", "vest")])[0]
    assert complete.worn == ("helmet", "vest")
    assert complete.missing == ()
    assert complete.is_compliant

    helmet_only = engine.evaluate([person], detected, [("helmet",)])[0]
    assert helmet_only.required_ppe == ("helmet",)
    assert helmet_only.worn == ("helmet",)
    assert helmet_only.missing == ()
    assert helmet_only.is_compliant
    assert "vest" not in helmet_only.worn + helmet_only.missing

    helmet_and_gloves = engine.evaluate([person], [helmet, vest], [("helmet", "gloves")])[0]
    assert helmet_and_gloves.worn == ("helmet",)
    assert helmet_and_gloves.missing == ("gloves",)
    assert "vest" not in helmet_and_gloves.worn + helmet_and_gloves.missing
    assert not helmet_and_gloves.is_compliant

    both_missing = engine.evaluate([person], [], [("helmet", "vest")])[0]
    assert both_missing.worn == ()
    assert both_missing.missing == ("helmet", "vest")
    assert not both_missing.is_compliant


def test_each_zone_checklist_is_applied_to_its_own_worker(settings: dict) -> None:
    engine = ComplianceEngine(settings)
    left_person = _box("person", (0.0, 0.0, 100.0, 300.0))
    right_person = _box("person", (200.0, 0.0, 300.0, 300.0))
    left = Zone("left", "Left", (Point(0, 0), Point(1, 0), Point(1, 1)), ("helmet",), 1)
    right = Zone("right", "Right", (Point(0, 0), Point(1, 0), Point(1, 1)), ("gloves", "vest"), 1)
    judged = engine.evaluate(
        [left_person, right_person],
        [
            _box("helmet", (40.0, 10.0, 60.0, 40.0)),
            _box("vest", (40.0, 120.0, 70.0, 180.0)),
            _box("vest", (230.0, 120.0, 270.0, 180.0)),
            _box("gloves", (220.0, 160.0, 250.0, 220.0)),
            _box("helmet", (230.0, 10.0, 260.0, 40.0)),
        ],
        [engine.required_for(left), engine.required_for(right)],
    )
    assert judged[0].required_ppe == ("helmet",)
    assert judged[0].worn == ("helmet",)
    assert judged[0].missing == ()
    assert judged[0].is_compliant
    assert "vest" not in judged[0].worn + judged[0].missing
    assert judged[1].required_ppe == ("gloves", "vest")
    assert judged[1].worn == ("vest", "gloves")
    assert judged[1].missing == ()
    assert "helmet" not in judged[1].worn + judged[1].missing


def test_shipped_defaults_require_helmet_and_vest_only() -> None:
    engine = ComplianceEngine(load_settings(ROOT / "config" / "settings.yaml"))
    assert engine.required_for(None) == ("helmet", "vest")


def test_a_vest_overlapping_the_person_is_not_listed_as_missing() -> None:
    """A vest box on the worker is worn, so the alert does not call it missing."""
    person = (0.0, 0.0, 100.0, 100.0)
    vest = _box("vest", (20.0, 70.0, 80.0, 95.0))
    assert classes_on_person(person, [vest], ("helmet", "vest")) == ("vest",)
    worn, missing, compliant, seen = reconcile_worn(
        ("helmet", "vest", "gloves", "boots"),
        (),
        ("helmet", "vest", "gloves", "boots"),
        ("vest",),
    )
    assert worn == ("vest",)
    assert missing == ("helmet", "gloves", "boots")
    assert seen == ("vest",)
    assert not compliant
