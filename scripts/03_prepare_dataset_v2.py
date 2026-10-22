
from collections import defaultdict, Counter
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dataset" / "extracted_frames_v2"
OUTPUT = ROOT / "dataset" / "splits_v2"

SEED = 42
SPLITS = ("train", "validation", "test")
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

# Each connected pair must remain entirely within one split.
PAIR_GROUPS = [
    {"183", "253"},
    {"469", "481"},
    {"585", "599"},
    {"672", "720"},
    {"866", "878"},
]

def pair_group_for_ids(ids):
    matches = [
        i for i, group in enumerate(PAIR_GROUPS)
        if ids & group
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely assign IDs {ids} to a pair group"
        )
    return matches[0]

def inspect_sources():
    real_dir = SOURCE / "real"
    fake_dir = SOURCE / "fake"

    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise FileNotFoundError(
            f"Expected source folders under {SOURCE}"
        )

    groups = defaultdict(lambda: defaultdict(list))
    counts = Counter()

    for path in real_dir.glob("*.jpg"):
        parts = path.stem.split("__")
        if len(parts) != 3 or parts[0] != "real":
            raise ValueError(f"Unexpected real filename: {path.name}")

        source_id = parts[1]
        group_id = pair_group_for_ids({source_id})
        groups[group_id]["real"].append(path)
        counts["real"] += 1

    for path in fake_dir.glob("*.jpg"):
        parts = path.stem.split("__")
        if len(parts) != 4 or parts[0] != "fake":
            raise ValueError(f"Unexpected fake filename: {path.name}")

        method, pair = parts[1], parts[2]
        if method not in METHODS:
            raise ValueError(f"Unknown method in {path.name}")

        ids = set(pair.split("_"))
        if len(ids) != 2:
            raise ValueError(f"Unexpected pair ID in {path.name}")

        group_id = pair_group_for_ids(ids)
        groups[group_id][f"fake:{method}"].append(path)
        counts[f"fake:{method}"] += 1

    if set(groups) != set(range(len(PAIR_GROUPS))):
        raise ValueError(
            f"Expected all five groups; found {sorted(groups)}"
        )

    return groups, counts

def main():
    groups, counts = inspect_sources()

    print("Source image counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    print("\nPair-group coverage:")
    for i, group in enumerate(PAIR_GROUPS):
        print(f"  Group {i}: {sorted(group)}")

    print(
        "\nWARNING: five independent groups are too few for a "
        "reliable three-way benchmark."
    )

    # Do not overwrite previous output.
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Inspect it before proceeding."
        )

    # Fixed group assignment, approximately 60/20/20 by group.
    rng = random.Random(SEED)
    group_ids = list(range(len(PAIR_GROUPS)))
    rng.shuffle(group_ids)

    assignment = {
        "train": group_ids[:3],
        "validation": group_ids[3:4],
        "test": group_ids[4:],
    }

    print("\nPlanned group assignment:")
    for split, ids in assignment.items():
        print(
            f"  {split}: "
            f"{[sorted(PAIR_GROUPS[i]) for i in ids]}"
        )

    # Copy only after all filename and group checks pass.
    for split in SPLITS:
        for cls in ("real", "fake"):
            (OUTPUT / split / cls).mkdir(
                parents=True, exist_ok=True
            )

    frame_counts = Counter()

    for split, group_ids in assignment.items():
        for group_id in group_ids:
            for label, paths in groups[group_id].items():
                cls = "real" if label == "real" else "fake"
                dest = OUTPUT / split / cls

                for path in paths:
                    target = dest / path.name
                    if target.exists():
                        raise FileExistsError(target)
                    shutil.copy2(path, target)
                    frame_counts[(split, cls)] += 1

    print("\nCopied image counts:")
    for split in SPLITS:
        print(
            f"  {split}: real={frame_counts[(split, 'real')]}, "
            f"fake={frame_counts[(split, 'fake')]}"
        )

    print(f"\nNew splits created at: {OUTPUT}")
    print("No model files were modified.")
    print("Training has not been started.")

if __name__ == "__main__":
    main()
