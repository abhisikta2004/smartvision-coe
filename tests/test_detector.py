"""Tests for the two-model PPE detector. No checkpoint is loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartvision.detector import PPEDetector, class_threshold


class FakeBoxes:
    """Stand-in for ultralytics boxes."""

    def __init__(self, rows: list[tuple[int, float, tuple[float, float, float, float]]]) -> None:
        if rows:
            self.xyxy = np.array([row[2] for row in rows], dtype=float)
            self.conf = np.array([row[1] for row in rows], dtype=float)
            self.cls = np.array([row[0] for row in rows], dtype=float)
        else:
            self.xyxy = np.zeros((0, 4))
            self.conf = np.zeros((0,))
            self.cls = np.zeros((0,))


class FakeResult:
    """One predict() result."""

    def __init__(
        self,
        rows: list[tuple[int, float, tuple[float, float, float, float]]],
        names: dict[int, str],
    ) -> None:
        self.names = names
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """Records predict() calls and returns a fixed set of boxes."""

    def __init__(
        self,
        rows: list[tuple[int, float, tuple[float, float, float, float]]],
        names: dict[int, str],
    ) -> None:
        self.names = names
        self.rows = rows
        self.calls = 0
        self.kwargs: list[dict] = []

    def predict(self, frame: np.ndarray, **kwargs: object) -> list[FakeResult]:
        self.calls += 1
        self.kwargs.append(kwargs)
        return [FakeResult(self.rows, self.names)]


PPE_NAMES = {0: "person", 1: "helmet", 2: "vest", 3: "gloves", 4: "boots", 5: "goggles"}
COCO_NAMES = {0: "person", 1: "bicycle"}


def test_missing_weights_raise_without_a_fallback(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    settings["model"]["weights"] = str(tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="no fallback"):
        PPEDetector(settings, project_root=tmp_path)
    del frame


def test_coco_mode_merges_person_and_ppe_and_warms_both_models(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not a real checkpoint")
    settings["model"]["weights"] = str(weights)
    ppe = FakeModel(
        [
            (0, 0.99, (1, 1, 2, 2)),
            (1, 0.40, (0, 0, 4, 4)),
            (1, 0.39, (0, 0, 4, 4)),
            (3, 0.15, (0, 0, 4, 4)),
            (3, 0.14, (0, 0, 4, 4)),
        ],
        PPE_NAMES,
    )
    person = FakeModel([(0, 0.50, (10, 10, 20, 40)), (1, 0.90, (0, 0, 3, 3))], COCO_NAMES)
    detector = PPEDetector(settings, ppe_model=ppe, person_model=person, project_root=tmp_path)
    assert ppe.calls == 1
    assert person.calls == 1
    found = detector.detect(frame)
    labels = [(det.class_name, det.confidence) for det in found]
    assert ("person", 0.99) not in labels
    assert ("person", 0.50) in labels
    assert ("helmet", 0.40) in labels
    assert ("helmet", 0.39) not in labels
    assert ("gloves", 0.15) in labels
    assert ("gloves", 0.14) not in labels
    assert all(det.class_name != "bicycle" for det in found)
    assert person.kwargs[-1]["classes"] == [0]
    assert person.kwargs[-1]["device"] == "cpu"
    assert 0 not in ppe.kwargs[-1]["classes"]


def test_ppe_model_source_uses_the_ppe_checkpoint_for_people(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not a real checkpoint")
    settings["model"]["weights"] = str(weights)
    settings["model"]["person_source"] = "ppe_model"
    ppe = FakeModel(
        [
            (0, 0.36, (10, 10, 30, 50)),
            (0, 0.20, (10, 10, 30, 50)),
            (1, 0.50, (0, 0, 4, 4)),
        ],
        PPE_NAMES,
    )
    detector = PPEDetector(settings, ppe_model=ppe, project_root=tmp_path)
    found = detector.detect(frame)
    people = [det for det in found if det.class_name == "person"]
    assert [det.confidence for det in people] == [0.36]
    assert any(det.class_name == "helmet" for det in found)


def test_unknown_person_source_is_rejected(settings: dict, tmp_path: Path) -> None:
    settings["model"]["person_source"] = "both"
    settings["model"]["weights"] = str(tmp_path / "best.pt")
    with pytest.raises(ValueError, match="person_source"):
        PPEDetector(settings, ppe_model=FakeModel([], PPE_NAMES), project_root=tmp_path)


def test_set_thresholds_replaces_the_class_cutoff(frame: np.ndarray, settings: dict, tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not a real checkpoint")
    settings["model"]["weights"] = str(weights)
    ppe = FakeModel([(1, 0.20, (0.0, 0.0, 4.0, 4.0))], PPE_NAMES)
    person = FakeModel([], COCO_NAMES)
    detector = PPEDetector(settings, ppe_model=ppe, person_model=person, project_root=tmp_path)
    assert [det.class_name for det in detector.detect(frame)] == []
    detector.set_thresholds(
        confidence=0.1,
        person_conf=0.1,
        class_confidence={"helmet": 0.1},
    )
    found = detector.detect(frame)
    assert [(det.class_name, det.confidence) for det in found] == [("helmet", 0.2)]


def test_class_threshold_uses_the_override_then_the_global_floor() -> None:
    overrides = {"gloves": 0.15}
    assert class_threshold("gloves", overrides, 0.25) == 0.15
    assert class_threshold("helmet", overrides, 0.25) == 0.25


def test_auto_device_skips_a_broken_mps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import smartvision.detector as detector_module

    class _Mps:
        @staticmethod
        def is_built() -> bool:
            return True

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch:
        class backends:
            mps = _Mps()

        cuda = _Cuda()

        @staticmethod
        def zeros(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("The MPS backend is supported on MacOS 14.0+")

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    assert detector_module.resolve_device("auto") == "cuda"
    assert detector_module.resolve_device("cpu") == "cpu"
