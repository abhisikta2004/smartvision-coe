"""Remap and split checks on a tiny fake YOLO dataset. No download, no camera."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.prepare_dataset import build_merged_dataset, load_config, remap_yolo_label

CANONICAL = ["person", "helmet", "vest", "gloves", "boots", "goggles"]


def _write_source(
    root: Path,
    names: list[str],
    images: dict[str, tuple[bytes, str]],
) -> None:
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": names, "nc": len(names)}),
        encoding="utf-8",
    )
    for name, (payload, label) in images.items():
        (image_dir / name).write_bytes(payload)
        (label_dir / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")


def _config(tmp_path: Path) -> Path:
    payload = {
        "canonical_classes": CANONICAL,
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 7},
        "min_instances": 300,
        "datasets": [
            {
                "id": "site",
                "workspace": "local",
                "project": "site",
                "version": 1,
                "classes": {"Person": "person", "Hardhat": "helmet"},
            },
            {
                "id": "ppe",
                "workspace": "local",
                "project": "ppe",
                "version": 1,
                "classes": {"helmet": "helmet", "boots": "boots"},
            },
        ],
    }
    path = tmp_path / "dataset_classes.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _fake_raw(raw_root: Path) -> bytes:
    """Build 20 kept images, one duplicate, and one image with only a dropped class."""
    duplicate = b"same-image-bytes"
    site_images: dict[str, tuple[bytes, str]] = {}
    for index in range(10):
        payload = duplicate if index == 0 else f"site-{index}".encode()
        # class 0 Person, class 2 Cone (not mapped, must be dropped)
        site_images[f"site_{index}.jpg"] = (
            payload,
            f"0 0.{index + 1:02d}000 0.500000 0.200000 0.200000\n2 0.100000 0.100000 0.100000 0.100000\n",
        )
    _write_source(raw_root / "site", ["Person", "Hardhat", "Cone"], site_images)

    ppe_images: dict[str, tuple[bytes, str]] = {}
    for index in range(10):
        # class 0 helmet -> canonical 1, class 1 boots -> canonical 4, class 2 cat dropped
        ppe_images[f"ppe_{index}.jpg"] = (
            f"ppe-{index}".encode(),
            "0 0.200000 0.200000 0.100000 0.100000\n"
            "1 0.300000 0.300000 0.100000 0.100000\n"
            "2 0.900000 0.900000 0.100000 0.100000\n",
        )
    ppe_images["duplicate.jpg"] = (duplicate, "0 0.990000 0.990000 0.100000 0.100000\n")
    ppe_images["only_cat.jpg"] = (b"cat-only", "2 0.400000 0.400000 0.100000 0.100000\n")
    _write_source(raw_root / "ppe", ["helmet", "boots", "cat"], ppe_images)
    return duplicate


def _labels(merged: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in (merged / "labels").rglob("*.txt"):
        found[path.stem] = path.read_text(encoding="utf-8")
    return found


def test_remap_drops_unmapped_classes() -> None:
    remapped = remap_yolo_label(
            "0 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.1 0.1\n3 0.3 0.3 0.2 0.2\n",
        ["helmet", "vest", "no-helmet", "goggles", "cat"],
        {"helmet": "helmet", "goggles": "goggles"},
        CANONICAL,
    )
    assert remapped.splitlines() == [
        "1 0.500000 0.500000 0.200000 0.200000",
        "5 0.300000 0.300000 0.200000 0.200000",
    ]


def test_fake_dataset_remaps_dedupes_and_splits(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _fake_raw(raw_root)
    config = load_config(_config(tmp_path))
    first = tmp_path / "merged-a"
    second = tmp_path / "merged-b"

    stats = build_merged_dataset(config, raw_root, first)
    build_merged_dataset(config, raw_root, second)

    assert stats.kept == 20
    assert stats.dropped_duplicate == 1
    assert stats.dropped_empty == 1
    assert stats.split_images == {"train": 16, "val": 2, "test": 2}
    assert sum(stats.totals) > 0

    labels = _labels(first)
    assert len(labels) == 20
    blob = "\n".join(labels.values())
    assert " 0.990000 " not in blob
    assert "0.900000" not in blob
    helmet_lines = [line for line in blob.splitlines() if line.startswith("1 ")]
    boots_lines = [line for line in blob.splitlines() if line.startswith("4 ")]
    person_lines = [line for line in blob.splitlines() if line.startswith("0 ")]
    assert len(helmet_lines) == 10
    assert len(boots_lines) == 10
    assert len(person_lines) == 10
    assert all(not line.startswith("2 ") for line in blob.splitlines())

    data = yaml.safe_load((first / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == CANONICAL
    assert data["nc"] == 6

    def stems(root: Path) -> dict[str, set[str]]:
        return {
            split: {path.stem for path in (root / "images" / split).iterdir()}
            for split in ("train", "val", "test")
        }

    assert stems(first) == stems(second)
    groups = stems(first)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
