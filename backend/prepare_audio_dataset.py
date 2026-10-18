
from pathlib import Path
import random
import os
import shutil

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "datasets" / "in_the_wild" / "release_in_the_wild"
METADATA = SOURCE_DIR / "meta.csv"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "audio"

SAMPLES_PER_CLASS = 2000
SEED = 42


def main():
    if not METADATA.is_file():
        raise FileNotFoundError(f"Metadata not found: {METADATA}")

    df = pd.read_csv(METADATA)

    required = {"file", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected CSV columns: {required}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    label_mapping = {
        "bona-fide": "real",
        "spoof": "fake",
    }

    df = df[df["label"].isin(label_mapping)].copy()
    rng = random.Random(SEED)

    for source_label, target_label in label_mapping.items():
        rows = df[df["label"] == source_label]
        rows = rows.sample(frac=1, random_state=SEED)

        selected = rows.head(SAMPLES_PER_CLASS)
        output_folder = OUTPUT_DIR / target_label
        output_folder.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0

        for _, row in selected.iterrows():
            filename = Path(str(row["file"])).name
            source = SOURCE_DIR / str(row["file"])
            destination = output_folder / filename

            # Prevent paths in metadata from escaping the dataset folder.
            if not source.resolve().is_relative_to(SOURCE_DIR.resolve()):
                print(f"Skipping unsafe path: {row['file']}")
                continue

            if not source.is_file():
                missing += 1
                continue

            if destination.exists():
                copied += 1
                continue

            try:
                # Hard links save disk space when both folders are on
                # the same drive. Fall back to copying if unavailable.
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

            copied += 1

        print(
            f"{target_label}: {copied} files available; "
            f"{missing} missing."
        )

    print(f"\nDataset prepared at: {OUTPUT_DIR}")
    print("Check both folders before training.")


if __name__ == "__main__":
    main()
