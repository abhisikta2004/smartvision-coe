"""Tests for the 5-second live safety panel. No camera and no model."""

from __future__ import annotations

from smartvision.live_status import StatusBoard, ppe_phrase, refresh_board
from smartvision.pipeline import Worker, WorkerStatus
from smartvision.zones import Point, Zone

LABELS = {
    "helmet": "Helmet",
    "vest": "Safety Vest",
    "gloves": "Gloves",
    "goggles": "Goggles",
    "boots": "Safety Boots",
}


def _zone() -> Zone:
    return Zone(
        zone_id="machine_1",
        name="Machine 1",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)),
        required_ppe=("helmet", "vest"),
        risk_level=1,
    )


def _worker(
    track_id: int,
    *,
    zone: Zone | None,
    missing: tuple[str, ...] = (),
    worn: tuple[str, ...] = (),
    uncertain: tuple[str, ...] = (),
) -> Worker:
    return Worker(
        track_id=track_id,
        bbox_xyxy=(0.0, 0.0, 10.0, 20.0),
        zone=zone,
        missing_ppe=missing,
        worn_ppe=worn,
        status=WorkerStatus.OK if not missing else WorkerStatus.CRITICAL,
        is_compliant=not missing,
        uncertain_ppe=uncertain,
    )


def test_ppe_phrase_joins_two_and_three_items() -> None:
    assert ppe_phrase(("helmet",), LABELS) == "Helmet"
    assert ppe_phrase(("helmet", "vest"), LABELS) == "Helmet and Safety Vest"
    assert ppe_phrase(("helmet", "vest", "boots"), LABELS) == "Helmet, Safety Vest, and Safety Boots"


def test_empty_frame_is_distinct_from_people_standing_outside() -> None:
    zone = _zone()
    empty = refresh_board(StatusBoard(), [], [zone], now=0.0, interval=5.0, labels=LABELS)
    assert empty.lines[0].message == "[SAFE] No person detected in the camera view."
    outside = refresh_board(
        StatusBoard(),
        [_worker(1, zone=None)],
        [zone],
        now=0.0,
        interval=5.0,
        labels=LABELS,
    )
    assert outside.lines[0].message == "[SAFE] No person detected inside Machine 1."
    assert outside.lines[0].kind == "safe"


def test_compliant_and_missing_ppe_are_reported_for_the_right_items() -> None:
    zone = _zone()
    board = refresh_board(
        StatusBoard(),
        [
            _worker(1, zone=zone, worn=("helmet", "vest")),
            _worker(2, zone=zone, worn=("helmet",), missing=("vest",)),
            _worker(3, zone=zone, missing=("helmet", "vest")),
        ],
        [zone],
        now=0.0,
        interval=5.0,
        labels=LABELS,
    )
    messages = [line.message for line in board.lines]
    assert messages[0] == (
        "[COMPLIANT] Person 1 is inside Machine 1 and is wearing all required PPE: Helmet and Safety Vest."
    )
    assert messages[1] == (
        "[SAFETY VIOLATION] Person 2 has entered Machine 1. "
        "Required PPE: Helmet and Safety Vest. Missing PPE: Safety Vest. "
        "The worker must wear the required safety vest while inside this zone. "
        "Detected PPE: Helmet."
    )
    assert "Helmet and Safety Vest" in messages[2]
    assert "Missing PPE: Helmet and Safety Vest." in messages[2]
    assert "has entered" not in messages[2]
    assert [line.track_id for line in board.lines] == [1, 2, 3]
    assert board.lines[1].event_id == 1
    assert board.lines[2].event_id == 2


def test_status_waits_five_seconds_and_does_not_duplicate_a_continuing_violation() -> None:
    zone = _zone()
    violator = _worker(2, zone=zone, worn=("helmet",), missing=("vest",))
    board = refresh_board(StatusBoard(), [violator], [zone], now=0.0, interval=5.0, labels=LABELS)
    early = refresh_board(
        board,
        [_worker(9, zone=zone, missing=("helmet", "vest"))],
        [zone],
        now=4.9,
        interval=5.0,
        labels=LABELS,
    )
    assert early is board
    assert early.lines[0].track_id == 2
    continued = refresh_board(board, [violator], [zone], now=5.0, interval=5.0, labels=LABELS)
    assert continued is not board
    assert continued.lines[0].event_id == board.lines[0].event_id
    assert "is inside" in continued.lines[0].message
    assert continued.lines[0].updated_at == 5.0
    clear = refresh_board(
        continued,
        [_worker(2, zone=zone, worn=("helmet", "vest"))],
        [zone],
        now=10.0,
        interval=5.0,
        labels=LABELS,
    )
    assert clear.lines[0].kind == "compliant"
    again = refresh_board(clear, [violator], [zone], now=15.0, interval=5.0, labels=LABELS)
    assert again.lines[0].kind == "violation"
    assert again.lines[0].event_id == board.lines[0].event_id + 1


def test_unselected_ppe_stays_out_of_the_alert() -> None:
    zone = Zone(
        zone_id="helmet_only",
        name="Helmet Bay",
        polygon=(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0)),
        required_ppe=("helmet",),
        risk_level=1,
    )
    board = refresh_board(
        StatusBoard(),
        [_worker(1, zone=zone, worn=("helmet", "vest", "gloves"))],
        [zone],
        now=0.0,
        interval=5.0,
        labels=LABELS,
    )
    line = board.lines[0]
    assert line.kind == "compliant"
    assert line.required_ppe == ("helmet",)
    assert line.detected_ppe == ("helmet",)
    assert line.missing_ppe == ()
    assert "Safety Vest" not in line.message
    assert "Gloves" not in line.message


def test_uncertain_ppe_is_not_called_missing() -> None:
    zone = _zone()
    board = refresh_board(
        StatusBoard(),
        [_worker(1, zone=zone, worn=("vest",), uncertain=("helmet",))],
        [zone],
        now=1.0,
        interval=5.0,
        labels=LABELS,
    )
    line = board.lines[0]
    assert line.kind == "uncertain"
    assert line.missing_ppe == ()
    assert "Could not verify: Helmet." in line.message
    assert "SAFETY VIOLATION" not in line.message
