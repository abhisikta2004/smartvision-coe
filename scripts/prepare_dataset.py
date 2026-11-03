"""Download PPE datasets, remap them to six classes, and merge a YOLO split.

The Roboflow API key is read from the environment (``.env`` key
``ROBOFLOW_API_KEY``). Class names and the split seed live in
``config/dataset_classes.yaml``. This script does not train a model.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SourceDataset:
    """One Universe project and the raw names that should be kept."""

    id: str
    workspace: str
    project: str
    version: int
    classes: dict[str, str]
    license: str = ""
    images: int | None = None
    url: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    """Canonical classes, split policy, and the datasets to merge."""

    canonical_classes: tuple[str, ...]
    datasets: tuple[SourceDataset, ...]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    min_instances: int


@dataclass
class Sample:
    """One kept image after remapping and hash deduplication."""

    digest: str
    image_path: Path
    label_text: str
    dataset_id: str


@dataclass
class MergeStats:
    """Counts produced by a merge, including per-split class totals."""

    images_seen: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0
    kept: int = 0
    split_images: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, list[int]] = field(default_factory=dict)
    canonical_classes: tuple[str, ...] = ()
    min_instances: int = 300

    @property
    def totals(self) -> list[int]:
        """Instance counts summed across train, val, and test."""
        width = len(self.canonical_classes)
        totals = [0] * width
        for counts in self.split_counts.values():
            for index, count in enumerate(counts):
                totals[index] += count
        return totals

    @property
    def flags(self) -> list[str]:
        """Classes whose total instance count is under ``min_instances``."""
        flagged: list[str] = []
        for name, count in zip(self.canonical_classes, self.totals):
            if count < self.min_instances:
                flagged.append(
                    f"FLAG: {name} has {count} instances, under {self.min_instances}"
                )
        return flagged


def load_config(path: Path) -> DatasetConfig:
    """Load the class map and split settings.

    Args:
        path: YAML file such as ``config/dataset_classes.yaml``.

    Returns:
        Parsed config. Mapped targets must be canonical class names.

    Raises:
        ValueError: The file is missing a required field, a mapped class is
            not canonical, or the split ratios do not sum to 1.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    canonical = tuple(raw["canonical_classes"])
    if len(canonical) != len(set(canonical)):
        raise ValueError("canonical_classes contains duplicates")
    split = raw["split"]
    train = float(split["train"])
    val = float(split["val"])
    test = float(split["test"])
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")

    datasets: list[SourceDataset] = []
    for entry in raw["datasets"]:
        classes = {str(source): str(target) for source, target in entry["classes"].items()}
        unknown = sorted(set(classes.values()) - set(canonical))
        if unknown:
            raise ValueError(f"{entry['id']} maps to unknown classes: {unknown}")
        datasets.append(
            SourceDataset(
                id=str(entry["id"]),
                workspace=str(entry["workspace"]),
                project=str(entry["project"]),
                version=int(entry["version"]),
                classes=classes,
                license=str(entry.get("license", "")),
                images=entry.get("images"),
                url=str(entry.get("url", "")),
            )
        )
    if not datasets:
        raise ValueError("datasets is empty")
    return DatasetConfig(
        canonical_classes=canonical,
        datasets=tuple(datasets),
        train_ratio=train,
        val_ratio=val,
        test_ratio=test,
        seed=int(split["seed"]),
        min_instances=int(raw["min_instances"]),
    )


def load_class_names(data_yaml: Path) -> list[str]:
    """Return class names from a YOLO ``data.yaml`` in index order.

    Args:
        data_yaml: A Roboflow or Ultralytics data file.

    Returns:
        Names keyed by class index. Dict-style ``names`` are sorted by index.
    """
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def remap_yolo_label(
    text: str,
    source_names: list[str],
    raw_to_canonical: dict[str, str],
    canonical: Iterable[str],
) -> str:
    """Rewrite one YOLO label file onto the canonical class indices.

    Each kept line is ``class_id x_center y_center width height``. Boxes whose
    source name is absent from ``raw_to_canonical`` are removed. Geometry is
    copied unchanged.

    Args:
        text: Original label file contents.
        source_names: Class names in the source dataset, index-aligned.
        raw_to_canonical: Source name to canonical name.
        canonical: Canonical names in the index order used in the merged set.

    Returns:
        Remapped label text. Empty when every box was dropped.
    """
    canonical_index = {name: index for index, name in enumerate(canonical)}
    kept: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            source_id = int(parts[0])
            coords = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if source_id < 0 or source_id >= len(source_names):
            continue
        target_name = raw_to_canonical.get(source_names[source_id])
        if target_name is None:
            continue
        target_id = canonical_index[target_name]
        geometry = " ".join(f"{value:.6f}" for value in coords)
        kept.append(f"{target_id} {geometry}")
    if not kept:
        return ""
    return "\n".join(kept) + "\n"


def split_image_ids(
    image_ids: Iterable[str],
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split image ids  into train, val, and test.

    Ids are sorted, then shuffled with ``seed``, so the same set always
    produces the same split. The val and test sizes are ``int(n * ratio)``.
    The remainder stays in train, which keeps every image in exactly one split.

    Args:
        image_ids: Unique image identifiers.
        train: Train fraction. Used to check that the three ratios sum to 1.
        val: Validation fraction.
        test: Test fraction.
        seed: Random seed.

    Returns:
        Mapping of ``train``, ``val``, and ``test`` to id lists.

    Raises:
        ValueError: The ratios do not sum to 1.
    """
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")
    ordered = sorted(image_ids)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    n_val = int(count * val)
    n_test = int(count * test)
    n_train = count - n_val - n_test
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def sha256_file(path: Path) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_data_yaml(root: Path) -> Path:
    """Locate the YOLO data file written by a dataset download.

    Args:
        root: Directory that contains, or contains a child with, ``data.yaml``.

    Returns:
        Path to the single ``data.yaml``.

    Raises:
        FileNotFoundError: None or more than one data file is present.
    """
    direct = root / "data.yaml"
    if direct.is_file():
        return direct
    found = [path for path in root.rglob("data.yaml") if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one data.yaml under {root}, found {len(found)}")
    return found[0]


def label_path_for(image: Path) -> Path | None:
    """Find the YOLO label that belongs to an image, if it exists.

    Args:
        image: Image path. Roboflow layouts use ``images`` and ``labels`` siblings.

    Returns:
        The label path, or ``None`` when the image has no label file.
    """
    candidates = []
    if image.parent.name == "images":
        candidates.append(image.parent.parent / "labels" / f"{image.stem}.txt")
    candidates.append(image.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def iter_images(root: Path) -> list[Path]:
    """List image files under a downloaded dataset.

    Args:
        root: Dataset root (the directory that holds ``data.yaml``).

    Returns:
        Image paths, sorted, skipping macOS metadata folders.
    """
    images: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("."):
            continue
        images.append(path)
    return sorted(images)


def collect_samples(config: DatasetConfig, raw_root: Path) -> tuple[list[Sample], MergeStats]:
    """Remap every downloaded dataset and drop empty or duplicate images.

    Datasets are visited in config order. The first copy of a file hash is kept,
    so an earlier dataset wins when the same image appears twice.

    Args:
        config: Class map and dataset list.
        raw_root: Directory of downloaded datasets, one subdirectory per id.

    Returns:
        Kept samples and partial stats (split counts are filled in later).

    Raises:
        RuntimeError: A mapped raw class is not in that dataset's ``data.yaml``.
    """
    stats = MergeStats(canonical_classes=config.canonical_classes, min_instances=config.min_instances)
    samples: dict[str, Sample] = {}
    for dataset in config.datasets:
        data_yaml = find_data_yaml(raw_root / dataset.id)
        source_names = load_class_names(data_yaml)
        missing = [name for name in dataset.classes if name not in source_names]
        if missing:
            raise RuntimeError(
                f"{dataset.id} is missing mapped classes {missing}. "
                f"data.yaml names are {source_names}"
            )
        for image in iter_images(data_yaml.parent):
            stats.images_seen += 1
            label_path = label_path_for(image)
            original = label_path.read_text(encoding="utf-8") if label_path else ""
            remapped = remap_yolo_label(
                original,
                source_names,
                dataset.classes,
                config.canonical_classes,
            )
            if not remapped:
                stats.dropped_empty += 1
                continue
            digest = sha256_file(image)
            if digest in samples:
                stats.dropped_duplicate += 1
                continue
            samples[digest] = Sample(
                digest=digest,
                image_path=image,
                label_text=remapped,
                dataset_id=dataset.id,
            )
    stats.kept = len(samples)
    return list(samples.values()), stats


def write_split(
    samples: list[Sample],
    config: DatasetConfig,
    output_dir: Path,
) -> dict[str, list[str]]:
    """Copy kept images into ``images/{split}`` and labels into ``labels/{split}``.

    Args:
        samples: Deduplicated, remapped samples.
        config: Split ratios and seed.
        output_dir: Merged dataset root. Replaced when it already exists.

    Returns:
        Split name to the digests assigned to that split.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    by_digest = {sample.digest: sample for sample in samples}
    assignment = split_image_ids(
        by_digest,
        train=config.train_ratio,
        val=config.val_ratio,
        test=config.test_ratio,
        seed=config.seed,
    )
    for split_name, digests in assignment.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for digest in digests:
            sample = by_digest[digest]
            stem = f"{sample.dataset_id}__{digest[:12]}"
            shutil.copy2(sample.image_path, image_dir / f"{stem}{sample.image_path.suffix.lower()}")
            (label_dir / f"{stem}.txt").write_text(sample.label_text, encoding="utf-8")
    _write_data_yaml(output_dir, config.canonical_classes)
    return assignment


def _write_data_yaml(output_dir: Path, canonical: tuple[str, ...] | list[str]) -> None:
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(list(canonical)),
        "names": list(canonical),
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def count_split(label_dir: Path, class_count: int) -> list[int]:
    """Count class instances in one split's label directory.

    Args:
        label_dir: Directory of YOLO ``.txt`` files.
        class_count: Number of canonical classes.

    Returns:
        Instance counts indexed by canonical class id.
    """
    counts = [0] * class_count
    if not label_dir.is_dir():
        return counts
    for path in sorted(label_dir.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            if 0 <= class_id < class_count:
                counts[class_id] += 1
    return counts


def build_merged_dataset(config: DatasetConfig, raw_root: Path, output_dir: Path) -> MergeStats:
    """Remap, deduplicate, split, and write the merged YOLO dataset.

    Args:
        config: Class map and split settings.
        raw_root: Downloaded datasets, one folder per dataset id.
        output_dir: Destination, replaced on each run.

    Returns:
        Image and per-class instance counts.
    """
    samples, stats = collect_samples(config, raw_root)
    write_split(samples, config, output_dir)
    for split_name in SPLIT_NAMES:
        counts = count_split(output_dir / "labels" / split_name, len(config.canonical_classes))
        stats.split_counts[split_name] = counts
        stats.split_images[split_name] = len(list((output_dir / "images" / split_name).glob("*")))
    return stats


def format_report(stats: MergeStats) -> str:
    """Format per-split counts and low-count flags.

    Args:
        stats: Merge results.

    Returns:
        A plain-text table. Classes under the threshold are marked ``FLAG``.
    """
    names = list(stats.canonical_classes)
    header = "split".ljust(8) + "images".rjust(8) + "".join(name.rjust(10) for name in names)
    lines = [
        f"seen={stats.images_seen} kept={stats.kept} "
        f"dropped_empty={stats.dropped_empty} dropped_duplicate={stats.dropped_duplicate}",
        header,
    ]
    for split_name in SPLIT_NAMES:
        counts = stats.split_counts.get(split_name, [0] * len(names))
        row = split_name.ljust(8) + str(stats.split_images.get(split_name, 0)).rjust(8)
        row += "".join(str(count).rjust(10) for count in counts)
        lines.append(row)
    totals = stats.totals
    total_images = sum(stats.split_images.get(split_name, 0) for split_name in SPLIT_NAMES)
    total_row = "TOTAL".ljust(8) + str(total_images).rjust(8)
    total_row += "".join(str(count).rjust(10) for count in totals)
    lines.append(total_row)
    flags = stats.flags
    if flags:
        lines.extend(flags)
    else:
        lines.append(f"No class is under {stats.min_instances} instances.")
    return "\n".join(lines)


def load_api_key() -> str:
    """Read ``ROBOFLOW_API_KEY`` from the environment or ``.env``.

    Returns:
        The API key.

    Raises:
        RuntimeError: The key is missing.
    """
    load_dotenv(ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Put it in .env and do not hardcode it.")
    return key


def download_datasets(config: DatasetConfig, raw_root: Path) -> None:
    """Download each configured dataset as YOLOv8 into ``raw_root/<id>``.

    A dataset is skipped when its folder already contains ``data.yaml``.
    Incomplete folders are removed and downloaded again.

    Args:
        config: Datasets to fetch.
        raw_root: Destination root, created if needed.
    """
    from roboflow import Roboflow

    raw_root.mkdir(parents=True, exist_ok=True)
    client = Roboflow(api_key=load_api_key())
    for dataset in config.datasets:
        destination = raw_root / dataset.id
        if find_existing_data_yaml(destination) is not None:
            print(f"skip download, already present: {dataset.id}")
            continue
        if destination.exists():
            shutil.rmtree(destination)
        print(f"downloading {dataset.workspace}/{dataset.project} v{dataset.version} -> {destination}")
        project = client.workspace(dataset.workspace).project(dataset.project)
        project.version(dataset.version).download("yolov8", location=str(destination))


def find_existing_data_yaml(root: Path) -> Path | None:
    """Return ``data.yaml`` if a previous download finished, else ``None``."""
    if not root.exists():
        return None
    try:
        return find_data_yaml(root)
    except FileNotFoundError:
        return None


def main() -> None:
    """Download, remap, merge, and print class counts."""
    config = load_config(ROOT / "config" / "dataset_classes.yaml")
    raw_root = ROOT / "data" / "raw"
    output_dir = ROOT / "data" / "merged"
    download_datasets(config, raw_root)
    stats = build_merged_dataset(config, raw_root, output_dir)
    print(format_report(stats))
    print(f"wrote {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
