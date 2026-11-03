"""Tests for dataset-path resolution, device choice, and eval tables."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smartvision.training import (
    coerce_fraction,
    epoch_seconds_from_results,
    estimate_full_seconds,
    render_markdown_table,
    rows_from_val_summary,
    select_device,
    subset_image_count,
    training_settings,
    write_absolute_data_yaml,
    write_eval_csv,
)


def test_select_device_prefers_mps_then_cuda_then_cpu() -> None:
    assert select_device("auto", mps_available=True, cuda_available=True) == "mps"
    assert select_device("auto", mps_available=False, cuda_available=True) == "cuda"
    assert select_device("auto", mps_available=False, cuda_available=False) == "cpu"
    assert select_device("cpu", mps_available=True, cuda_available=True) == "cpu"


def test_write_absolute_data_yaml_resolves_dot_from_the_file(tmp_path: Path) -> None:
    dataset = tmp_path / "merged"
    dataset.mkdir()
    source = dataset / "data.yaml"
    source.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n- person\n",
        encoding="utf-8",
    )
    destination = tmp_path / "runs" / "_data.yaml"
    write_absolute_data_yaml(source, destination)
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert Path(payload["path"]) == dataset.resolve()
    assert payload["train"] == "images/train"
    assert payload["names"] == ["person"]


def test_fraction_counts_and_full_run_estimate() -> None:
    assert coerce_fraction([200, 40]) == [200, 40]
    assert subset_image_count(4450, [200, 40], 0) == 200
    assert subset_image_count(556, [200, 40], 1) == 40
    seconds = estimate_full_seconds(
        epoch_seconds=12,
        smoke_images=240,
        full_images=5006,
        smoke_epochs=1,
        full_epochs=50,
    )
    assert seconds == pytest.approx(12 / 240 * 5006 * 50)


def test_epoch_seconds_are_deltas(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("epoch,time,loss\n1,10,1\n2,25,0.5\n", encoding="utf-8")
    assert epoch_seconds_from_results(path) == [10, 15]


def test_eval_table_round_trip(tmp_path: Path) -> None:
    rows = rows_from_val_summary(
        [
            {
                "Class": "helmet",
                "Box-P": 0.5,
                "Box-R": 0.25,
                "mAP50": 0.4,
                "mAP50-95": 0.2,
            }
        ]
    )
    csv_path = tmp_path / "eval_results.csv"
    write_eval_csv(rows, csv_path)
    text = csv_path.read_text(encoding="utf-8")
    assert "helmet,0.5,0.25,0.4,0.2" in text
    table = render_markdown_table(rows)
    assert "| helmet | 0.5 | 0.25 | 0.4 | 0.2 |" in table


def test_training_settings_require_the_block() -> None:
    with pytest.raises(KeyError):
        training_settings({})
