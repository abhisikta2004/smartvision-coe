"""Pure helpers for YOLOv8 fine-tuning and evaluation.

No model is loaded here. Scripts call these functions, then hand the
resolved dataset file to ultralytics. Device policy is MPS, then CUDA, then CPU.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

TRAINING_KEYS = (
    "model",
    "data",
    "imgsz",
    "epochs",
    "batch",
    "patience",
    "device",
    "workers",
    "project",
    "name",
    "weights_out",
)


def training_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the training block from a parsed settings file.

    Args:
        settings: Parsed ``config/settings.yaml``.

    Returns:
        The ``training`` mapping.

    Raises:
        KeyError: ``training`` is missing or a required key is absent.
    """
    block = settings.get("training")
    if not isinstance(block, dict):
        raise KeyError("config/settings.yaml is missing a training: block")
    missing = [key for key in TRAINING_KEYS if key not in block]
    if missing:
        raise KeyError(f"training settings missing {missing}")
    return dict(block)


def select_device(requested: str, *, mps_available: bool, cuda_available: bool) -> str:
    """Choose a torch device.

    ``auto`` tries MPS, then CUDA, then CPU. Any other value is returned
    unchanged so a CLI override is honored.

    Args:
        requested: ``auto`` or an explicit device string.
        mps_available: Whether this Mac can use MPS.
        cuda_available: Whether CUDA is available.

    Returns:
        Device string understood by ultralytics (``mps``, ``cuda``, or ``cpu``,
        or the explicit override).
    """
    choice = requested.strip().lower()
    if choice != "auto":
        return requested.strip()
    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def write_absolute_data_yaml(source: Path, destination: Path) -> Path:
    """Write a copy of a YOLO data file with ``path`` made absolute.

    Relative ``path`` values, including ``.``, are resolved from the directory
    that contains ``source``, not from the process working directory. Train,
    val, and test entries are left relative to that dataset root.

    Args:
        source: Existing data yaml, usually ``data/merged/data.yaml``.
        destination: Where to write the temporary copy.

    Returns:
        ``destination``.
    """
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} did not contain a mapping")
    raw_path = payload.get("path", ".")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (source.parent / path).resolve()
    payload["path"] = str(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def coerce_fraction(values: Iterable[float] | None) -> int | float | list[int | float] | None:
    """Turn CLI fraction values into the form ultralytics treats as counts.

    A number greater than 1 that is a whole number becomes an ``int`` image
    count. A value in ``(0, 1]`` stays a ratio. One value is returned bare;
    several values become a ``[train, val, test]`` list.

    Args:
        values: Fraction tokens from the CLI, or ``None`` to use the full set.

    Returns:
        ``None``, a single count or ratio, or a per-split list.
    """
    if values is None:
        return None
    coerced: list[int | float] = []
    for value in values:
        if value > 1 and float(value).is_integer():
            coerced.append(int(value))
        else:
            coerced.append(float(value))
    if len(coerced) == 1:
        return coerced[0]
    return coerced


def subset_image_count(total: int, fraction: int | float | list[int | float] | None, split_index: int) -> int:
    """How many images a fraction keeps for one split.

    Args:
        total: Images available in that split.
        fraction: Value produced by :func:`coerce_fraction`.
        split_index: ``0`` train, ``1`` val, ``2`` test.

    Returns:
        The capped image count ultralytics will load.
    """
    if fraction is None or total <= 0:
        return total
    value: int | float
    if isinstance(fraction, list):
        value = fraction[split_index] if split_index < len(fraction) else 1
    else:
        value = fraction
    if isinstance(value, int) and not isinstance(value, bool) and value > 1:
        return min(total, value)
    return max(1, round(total * float(value))) if float(value) > 0 else 0


def epoch_seconds_from_results(results_csv: Path) -> list[float]:
    """Read per-epoch durations from an Ultralytics ``results.csv``.

    The ``time`` column is cumulative seconds from the start of the training
    loop. Durations are the differences between successive rows.

    Args:
        results_csv: CSV written by the trainer.

    Returns:
        One duration per recorded epoch. Empty when the file has no rows.
    """
    with results_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time" not in reader.fieldnames:
            raise ValueError(f"{results_csv} has no time column")
        cumulative = [float(row["time"]) for row in reader if row.get("time")]
    if not cumulative:
        return []
    durations = [cumulative[0]]
    for previous, current in zip(cumulative, cumulative[1:]):
        durations.append(current - previous)
    return durations


def estimate_full_seconds(
    epoch_seconds: float,
    smoke_images: int,
    full_images: int,
    smoke_epochs: int,
    full_epochs: int,
) -> float:
    """Scale a measured epoch to the full image count and epoch budget.

    Args:
        epoch_seconds: Seconds for ``smoke_epochs`` on ``smoke_images``.
        smoke_images: Train plus val images used in the measured run.
        full_images: Train plus val images in the full dataset.
        smoke_epochs: Epochs that produced ``epoch_seconds``.
        full_epochs: Epochs the real run would use.

    Returns:
        Estimated seconds for the full run.

    Raises:
        ValueError: A count or the measured epoch time is not positive.
    """
    if epoch_seconds <= 0 or smoke_images <= 0 or smoke_epochs <= 0 or full_epochs <= 0:
        raise ValueError("epoch time, image counts, and epoch counts must be positive")
    per_image_epoch = epoch_seconds / (smoke_images * smoke_epochs)
    return per_image_epoch * full_images * full_epochs


def rows_from_val_summary(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map an ultralytics validation summary onto the eval table columns.

    Args:
        summary: Rows from ``DetMetrics.summary()``.

    Returns:
        Rows with ``class``, ``precision``, ``recall``, ``mAP50``, and ``mAP50-95``.
    """
    rows: list[dict[str, Any]] = []
    for item in summary:
        rows.append(
            {
                "class": item["Class"],
                "precision": item["Box-P"],
                "recall": item["Box-R"],
                "mAP50": item["mAP50"],
                "mAP50-95": item["mAP50-95"],
            }
        )
    return rows


def write_eval_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write per-class evaluation rows to CSV.

    Args:
        rows: Rows from :func:`rows_from_val_summary`.
        path: Destination CSV. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class", "precision", "recall", "mAP50", "mAP50-95"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render per-class metrics as a Markdown table.

    Args:
        rows: Rows with the eval columns.

    Returns:
        A Markdown table, including the header.
    """
    header = "| class | precision | recall | mAP50 | mAP50-95 |"
    separator = "| --- | ---: | ---: | ---: | ---: |"
    body = [
        f"| {row['class']} | {row['precision']} | {row['recall']} | {row['mAP50']} | {row['mAP50-95']} |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"
