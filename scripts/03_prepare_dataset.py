
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

SOURCE_REAL = DATASET_DIR / "extracted_frames" / "real"
SOURCE_FAKE = DATASET_DIR / "extracted_frames" / "fake"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-8

SPLITS = ("train", "validation", "test")
CLASSES = ("fake", "real")


def source_video_key(image_path, class_name):
    """
    Extract the source-video identity from the frame filename.

    Expected examples:
      real__video123__000001.jpg
      fake__Deepfakes__video123__000001.jpg

    The final numeric component is the extracted-frame index.
    """
    parts = image_path.stem.rsplit("__", 1)

    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Unexpected frame filename: {image_path.name}. "
            "Cannot safely identify its source video."
        )

    return parts[0]


def collect_groups(source_dir, class_name):
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    groups = defaultdict(list)

    for image_path in sorted(source_dir.glob("*.jpg")):
        key = source_video_key(image_path, class_name)
        groups[key].append(image_path)

    if not groups:
        raise ValueError(f"No JPG images found in {source_dir}")

    return groups


def split_groups(groups):
    """Split source videos, not individual frames."""
    video_keys = sorted(groups)

    if len(video_keys) < 3:
        raise ValueError(
            f"Need at least 3 source videos per class; found {len(video_keys)}."
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(video_keys)

    n = len(video_keys)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VALID_RATIO))

    # Ensure every split has at least one source video.
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    train_keys = video_keys[:n_train]
    val_keys = video_keys[n_train:n_train + n_val]
    test_keys = video_keys[n_train + n_val:]

    return {
        "train": train_keys,
        "validation": val_keys,
        "test": test_keys,
    }


def main():
    # Validate all source files before deleting any previous split.
    source_groups = {
        "real": collect_groups(SOURCE_REAL, "real"),
        "fake": collect_groups(SOURCE_FAKE, "fake"),
    }

    assignments = {
        class_name: split_groups(groups)
        for class_name, groups in source_groups.items()
    }

    # Remove stale copies from all prior splits.
    for split in SPLITS:
        for class_name in CLASSES:
            output_dir = DATASET_DIR / split / class_name
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreparing video-level dataset splits...\n")

    frame_counts = {
        split: {class_name: 0 for class_name in CLASSES}
        for split in SPLITS
    }

    for class_name, groups in source_groups.items():
        used_videos = set()

        for split, video_keys in assignments[class_name].items():
            output_dir = DATASET_DIR / split / class_name

            for video_key in video_keys:
                if video_key in used_videos:
                    raise RuntimeError(
                        f"Source video leakage detected: {video_key}"
                    )
                used_videos.add(video_key)

                for image_path in groups[video_key]:
                    shutil.copy2(image_path, output_dir / image_path.name)
                    frame_counts[split][class_name] += 1

        if used_videos != set(groups):
            raise RuntimeError(f"Some {class_name} videos were not assigned.")

    # Report actual frame and source-video counts.
    for split in SPLITS:
        print(f"\n{split.upper()}")
        for class_name in CLASSES:
            video_count = len(assignments[class_name][split])
            image_count = frame_counts[split][class_name]
            print(
                f"{class_name.capitalize():5} | "
                f"videos: {video_count:3} | frames: {image_count:5}"
            )

    total_frames = sum(
        frame_counts[split][class_name]
        for split in SPLITS
        for class_name in CLASSES
    )

    print(f"\nTotal assigned frames: {total_frames}")
    print("Video-level dataset preparation completed.")


if __name__ == "__main__":
    main()
