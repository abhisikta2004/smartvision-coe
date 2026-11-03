"""Shared fixtures. Frames are synthetic so tests never open a camera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def frame() -> np.ndarray:
    """Small BGR frame accepted by every stage that takes an image."""
    return np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.fixture
def settings(tmp_path: Path) -> dict:
    """Backend settings with outputs pointed at a temp directory."""
    return {
        "model": {
            "weights": str(tmp_path / "missing-best.pt"),
            "person_weights": "yolov8s.pt",
            "person_source": "coco",
            "person_coco_class": 0,
            "person_conf": 0.35,
            "confidence": 0.25,
            "iou": 0.45,
            "imgsz": 64,
            "device": "cpu",
            "class_confidence": {
                "helmet": 0.4,
                "vest": 0.4,
                "boots": 0.4,
                "goggles": 0.25,
                "gloves": 0.15,
            },
        },
        "classes": {
            "person": "person",
            "ppe": ["helmet", "vest", "gloves", "goggles", "boots"],
        },
        "compliance": {
            "default_required_ppe": ["helmet", "vest"],
            "violation_hold_seconds": 2.0,
            "body_regions": {
                "helmet": {"y_start": 0.0, "y_end": 0.333333},
                "goggles": {"y_start": 0.0, "y_end": 0.333333},
                "vest": {"y_start": 0.333333, "y_end": 0.666667},
                "gloves": {"y_start": 0.333333, "y_end": 1.0},
                "boots": {"y_start": 0.75, "y_end": 1.0},
            },
        },
        "tracking": {"iou_threshold": 0.3, "max_lost_frames": 1},
        "alerts": {
            "evidence_dir": str(tmp_path / "evidence"),
            "database": str(tmp_path / "alerts.db"),
            "cooldown_seconds": 10.0,
            "box_color": [0, 0, 220],
            "zone_color": [255, 128, 0],
        },
        "visualize": {
            "colors": {
                "ok": [0, 180, 0],
                "warning": [0, 165, 255],
                "critical": [0, 0, 220],
                "ppe": [255, 255, 0],
                "zone": [255, 128, 0],
            },
            "thickness": 2,
            "font_scale": 0.5,
        },
        "pipeline": {"frame_skip": 0},
        "review": {
            "auto_zone": {
                "name": "Work area",
                "risk": 1,
                "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            }
        },
    }
