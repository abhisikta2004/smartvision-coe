"""Fine-tune YOLOv8 on the merged PPE dataset.

Defaults come from ``config/settings.yaml`` and can be overridden on the
command line. The dataset yaml is rewritten with an absolute ``path`` so the
run does not depend on the working directory. A failed MPS run is retried
once on CPU.

Example:
    python scripts/train.py
    python scripts/train.py --epochs 1 --fraction 200 40 --name smoke --close-mosaic 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smartvision.training import (  # noqa: E402
    coerce_fraction,
    epoch_seconds_from_results,
    estimate_full_seconds,
    select_device,
    subset_image_count,
    training_settings,
    write_absolute_data_yaml,
)


def load_settings(path: Path) -> dict[str, Any]:
    """Load ``config/settings.yaml``.

    Args:
        path: Settings file.

    Returns:
        Parsed mapping.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return payload


def parse_args(argv: list[str] | None, defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse CLI overrides for the training settings.

    Args:
        argv: Arguments excluding the program name. ``None`` reads sys.argv.
        defaults: Training block from settings.

    Returns:
        Parsed arguments. Unset options stay ``None`` so the settings value is used.
    """
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on data/merged.")
    parser.add_argument("--model", default=None, help=f"checkpoint (default {defaults['model']})")
    parser.add_argument("--data", default=None, help=f"data yaml (default {defaults['data']})")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, mps, cuda, or cpu")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default=None, help="run root, default outputs/runs")
    parser.add_argument("--name", default=None, help="run name under the project directory")
    parser.add_argument("--weights-out", default=None, help="where to copy best.pt")
    parser.add_argument(
        "--fraction",
        nargs="+",
        type=float,
        default=None,
        help="image count or ratio; one value or train val [test]",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="disable mosaic for the last N epochs; 0 keeps mosaic on",
    )
    return parser.parse_args(argv)


def resolve_options(args: argparse.Namespace, defaults: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI overrides onto the settings defaults.

    Args:
        args: Parsed CLI.
        defaults: Training settings.

    Returns:
        Options for one training run. Paths are absolute.
    """
    options = dict(defaults)
    for key in ("model", "data", "imgsz", "epochs", "batch", "patience", "device", "workers", "project", "name"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    if args.weights_out is not None:
        options["weights_out"] = args.weights_out
    options["fraction"] = coerce_fraction(args.fraction)
    options["close_mosaic"] = args.close_mosaic
    options["data"] = str((ROOT / options["data"]).resolve() if not Path(options["data"]).is_absolute() else Path(options["data"]))
    options["project"] = str((ROOT / options["project"]).resolve() if not Path(options["project"]).is_absolute() else Path(options["project"]))
    options["weights_out"] = str(
        (ROOT / options["weights_out"]).resolve()
        if not Path(options["weights_out"]).is_absolute()
        else Path(options["weights_out"])
    )
    return options


def resolve_training_device(requested: str) -> tuple[str, str | None]:
    """Pick a device, probing MPS with a real op when it might be used.

    ``torch.backends.mps.is_available()`` is false on macOS versions this
    PyTorch build rejects. The probe surfaces that error so the run can fall
    back to CPU instead of failing mid-epoch.

    Args:
        requested: ``auto`` or an explicit device.

    Returns:
        The device to train on, and a fallback note when MPS was skipped.
    """
    choice = requested.strip().lower()
    wants_mps = choice in {"auto", "mps"}
    if wants_mps and torch.backends.mps.is_built():
        try:
            torch.zeros(1, device="mps")
            if choice == "auto":
                return "mps", None
        except Exception as exc:
            note = f"MPS failed ({exc.__class__.__name__}: {exc})."
            if choice == "mps":
                return "cpu", note
            return (
                select_device(
                    "auto",
                    mps_available=False,
                    cuda_available=torch.cuda.is_available(),
                ),
                note,
            )
    return (
        select_device(
            requested,
            mps_available=torch.backends.mps.is_available(),
            cuda_available=torch.cuda.is_available(),
        ),
        None,
    )


def run_training(options: dict[str, Any], device: str, data_yaml: Path) -> Path:
    """Train and return the run directory.

    Args:
        options: Resolved training options.
        device: Torch device string.
        data_yaml: Dataset yaml with an absolute path.

    Returns:
        Ultralytics save directory for this run.
    """
    from ultralytics import YOLO

    model = YOLO(options["model"])
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(options["epochs"]),
        "imgsz": int(options["imgsz"]),
        "batch": int(options["batch"]),
        "patience": int(options["patience"]),
        "device": device,
        "workers": int(options["workers"]),
        "project": options["project"],
        "name": options["name"],
        "exist_ok": True,
        "pretrained": True,
    }
    if options["fraction"] is not None:
        train_kwargs["fraction"] = options["fraction"]
    if options["close_mosaic"] is not None:
        train_kwargs["close_mosaic"] = int(options["close_mosaic"])
    model.train(**train_kwargs)
    save_dir = Path(model.trainer.save_dir)
    return save_dir


def copy_best(save_dir: Path, weights_out: Path) -> Path:
    """Copy ``weights/best.pt`` to the configured output path.

    Args:
        save_dir: Ultralytics run directory.
        weights_out: Destination checkpoint.

    Returns:
        The destination path.

    Raises:
        FileNotFoundError: The run did not save ``best.pt``.
    """
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished without {best}")
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, weights_out)
    return weights_out


def count_images(directory: Path) -> int:
    """Count image files directly under a split directory.

    Args:
        directory: ``images/train`` or ``images/val``.

    Returns:
        Number of files. Zero when the directory is missing.
    """
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file())


def print_estimate(options: dict[str, Any], save_dir: Path, device: str, fell_back: bool) -> None:
    """Print the measured epoch time and a full-run estimate.

    Args:
        options: Resolved options, including the settings epoch budget in
            ``full_epochs`` when this invocation overrode ``epochs``.
        save_dir: Run directory that contains ``results.csv``.
        device: Device that actually trained.
        fell_back: Whether MPS failed and this run used CPU.
    """
    results_csv = save_dir / "results.csv"
    durations = epoch_seconds_from_results(results_csv)
    measured = sum(durations)
    data_yaml = Path(options["data"])
    dataset_root = data_yaml.parent
    full_train = count_images(dataset_root / "images" / "train")
    full_val = count_images(dataset_root / "images" / "val")
    smoke_train = subset_image_count(full_train, options["fraction"], 0)
    smoke_val = subset_image_count(full_val, options["fraction"], 1)
    full_epochs = int(options["full_epochs"])
    estimate = estimate_full_seconds(
        measured,
        smoke_train + smoke_val,
        full_train + full_val,
        max(len(durations), 1),
        full_epochs,
    )
    hours = estimate / 3600
    print(
        f"device={device}"
        + (" (fell back from mps)" if fell_back else "")
        + f" measured_epochs={len(durations)} measured_seconds={measured:.1f}"
    )
    print(
        f"measured images train={smoke_train} val={smoke_val}; "
        f"full images train={full_train} val={full_val}"
    )
    print(f"estimated full run at {full_epochs} epochs: {hours:.2f} hours ({estimate:.0f} s)")


def main(argv: list[str] | None = None) -> None:
    """Train, copy ``best.pt``, and print a full-run time estimate."""
    settings = load_settings(ROOT / "config" / "settings.yaml")
    defaults = training_settings(settings)
    args = parse_args(argv, defaults)
    options = resolve_options(args, defaults)
    options["full_epochs"] = int(defaults["epochs"])
    requested, mps_note = resolve_training_device(str(options["device"]))
    if mps_note:
        print(mps_note + f" Using {requested} for this run.")
    source = Path(options["data"])
    resolved = write_absolute_data_yaml(source, Path(options["project"]) / "_data.yaml")
    print(f"resolved dataset path -> {resolved}")

    started = time.perf_counter()
    fell_back = False
    try:
        save_dir = run_training(options, requested, resolved)
        device_used = requested
    except Exception as exc:
        if requested != "mps":
            raise
        fell_back = True
        print(f"MPS training failed ({exc.__class__.__name__}: {exc}). Retrying on cpu.")
        save_dir = run_training(options, "cpu", resolved)
        device_used = "cpu"
    copied = copy_best(save_dir, Path(options["weights_out"]))
    elapsed = time.perf_counter() - started
    print(f"copied {copied}")
    print(f"wall_seconds={elapsed:.1f}")
    print_estimate(options, save_dir, device_used, fell_back)


if __name__ == "__main__":
    main()
