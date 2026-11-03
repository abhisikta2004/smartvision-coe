"""Run a video or webcam through the SmartVision pipeline.

The annotated video is written under ``outputs/``. Critical alerts are
printed as they are raised. Frame skipping uses ``--skip`` or
``pipeline.frame_skip`` in ``config/settings.yaml``.

Example:
    python scripts/run_video.py data/samples/site.mp4
    python scripts/run_video.py data/samples/site.mp4 --skip 2
    python scripts/run_video.py 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smartvision.pipeline import SmartVisionPipeline, open_video_capture  # noqa: E402


def output_path(source: str, root: Path) -> Path:
    """Choose the annotated video path for a source.

    Args:
        source: File path, URL, or webcam index string.
        root: Repository root.

    Returns:
        ``outputs/<stem>_annotated.mp4`` for a file, or a fixed name for
        a webcam or a stream.
    """
    path = Path(source)
    if path.suffix:
        name = f"{path.stem}_annotated.mp4"
    elif source.isdigit():
        name = "webcam_annotated.mp4"
    else:
        name = "annotated.mp4"
    return root / "outputs" / name


def run(
    source: str,
    pipeline: Any,
    output: Path,
    frame_skip: int | None,
) -> Path:
    """Process ``source`` and write the annotated video.

    Args:
        source: File path, RTSP URL, or webcam index string.
        pipeline: Object with ``process_video(source, frame_skip=...)``.
        output: Destination video path.
        frame_skip: Frames to skip between processed frames.

    Returns:
        ``output`` after the writer is released. When the source has no
        frames, the file is not created.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = open_video_capture(source if not source.isdigit() else int(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    writer: cv2.VideoWriter | None = None
    frames = 0
    critical = 0
    previous: dict[int, str] = {}
    try:
        for result in pipeline.process_video(source, frame_skip=frame_skip, capture=capture):
            annotated = result.annotated_frame
            if not isinstance(annotated, np.ndarray):
                raise RuntimeError("pipeline did not return an annotated frame")
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(annotated)
            frames += 1
            for worker in result.workers:
                status = getattr(worker.status, "value", worker.status)
                if status == "WARNING" and previous.get(worker.track_id) != "WARNING":
                    zone = worker.zone.name if worker.zone is not None else "outside"
                    missing = ", ".join(worker.missing_ppe) or "ppe"
                    print(
                        f"WARNING worker={worker.track_id} zone={zone} "
                        f"missing={missing} t={result.timestamp:.2f}"
                    )
                previous[worker.track_id] = str(status)
            for alert in result.alerts:
                critical += 1
                missing = ", ".join(alert.missing_ppe)
                print(
                    f"CRITICAL worker={alert.track_id} zone={alert.zone_name or alert.zone_id} "
                    f"missing={missing} t={alert.timestamp:.2f} evidence={alert.evidence_path}"
                )
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    print(f"frames={frames} critical_alerts={critical} output={output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the video source and the optional frame skip.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        Parsed namespace with ``source``, ``skip``, and ``output``.
    """
    parser = argparse.ArgumentParser(description="Run SmartVision on a video or webcam.")
    parser.add_argument("source", help="video path, rtsp url, or webcam index")
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help="frames to skip between processed frames (default: pipeline.frame_skip)",
    )
    parser.add_argument("--output", default=None, help="annotated mp4 path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load config, run the pipeline, and print alerts."""
    args = parse_args(argv)
    destination = Path(args.output) if args.output else output_path(args.source, ROOT)
    if not destination.is_absolute():
        destination = ROOT / destination
    pipeline = SmartVisionPipeline(project_root=ROOT)
    run(args.source, pipeline, destination, args.skip)


if __name__ == "__main__":
    main()
