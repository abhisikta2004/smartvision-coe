"""Evaluate a trained YOLO checkpoint on the merged test split.

Writes per-class precision, recall, mAP50, and mAP50-95 to
``outputs/eval_results.csv`` and prints a Markdown table.

Example:
    python scripts/evaluate.py --weights models/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.train import resolve_training_device  # noqa: E402
from smartvision.training import (  # noqa: E402
    render_markdown_table,
    rows_from_val_summary,
    training_settings,
    write_absolute_data_yaml,
    write_eval_csv,
)


def parse_args(argv: list[str] | None, defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse evaluation CLI overrides.

    Args:
        argv: Arguments excluding the program name.
        defaults: Training block, used for the dataset path and device.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate YOLO weights on the test split.")
    parser.add_argument("--weights", default=None, help=f"checkpoint (default {defaults['weights_out']})")
    parser.add_argument("--data", default=None, help=f"data yaml (default {defaults['data']})")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, mps, cuda, or cpu")
    parser.add_argument("--split", default="test")
    parser.add_argument("--csv", default="outputs/eval_results.csv")
    parser.add_argument("--markdown", default="outputs/eval_results.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run test-split validation and write the metric table."""
    from ultralytics import YOLO

    settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    defaults = training_settings(settings)
    args = parse_args(argv, defaults)
    weights = Path(args.weights or defaults["weights_out"])
    if not weights.is_absolute():
        weights = ROOT / weights
    data = Path(args.data or defaults["data"])
    if not data.is_absolute():
        data = ROOT / data
    device, mps_note = resolve_training_device(args.device or str(defaults["device"]))
    if mps_note:
        print(mps_note + f" Using {device} for this run.")
    resolved = write_absolute_data_yaml(data, ROOT / "outputs" / "runs" / "_data.yaml")
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(resolved),
        split=args.split,
        imgsz=args.imgsz or int(defaults["imgsz"]),
        batch=args.batch or int(defaults["batch"]),
        device=device,
        project=str(ROOT / "outputs" / "runs"),
        name="eval",
        exist_ok=True,
        plots=False,
    )
    rows = rows_from_val_summary(metrics.summary())
    csv_path = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    markdown_path = ROOT / args.markdown if not Path(args.markdown).is_absolute() else Path(args.markdown)
    write_eval_csv(rows, csv_path)
    table = render_markdown_table(rows)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
