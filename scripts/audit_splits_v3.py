import csv
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "dataset" / "splits_v3" / "manifest.csv"

if not MANIFEST.exists():
    raise SystemExit(f"STOP: manifest not found: {MANIFEST}")

with MANIFEST.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

required = {
    "image_path", "label", "method",
    "identity_group", "split", "source_version"
}

if not rows:
    raise SystemExit("STOP: manifest is empty.")

if not required.issubset(rows[0].keys()):
    raise SystemExit(f"STOP: missing columns: {required - set(rows[0].keys())}")

errors = []
seen_paths = set()
group_splits = defaultdict(set)
counts = Counter()

for row in rows:
    rel = row["image_path"]
    path = PROJECT / rel
    key = rel.lower()

    if key in seen_paths:
        errors.append(f"Duplicate manifest path: {rel}")
    seen_paths.add(key)

    if not path.is_file():
        errors.append(f"Missing image file: {rel}")

    if row["label"] not in {"real", "fake"}:
        errors.append(f"Invalid label: {rel} -> {row['label']}")

    if row["split"] not in {"train", "validation", "test"}:
        errors.append(f"Invalid split: {rel} -> {row['split']}")

    if not row["identity_group"]:
        errors.append(f"Missing identity group: {rel}")

    group_splits[row["identity_group"]].add(row["split"])
    counts[(row["split"], row["label"])] += 1

leaking_groups = {
    group: splits for group, splits in group_splits.items()
    if len(splits) != 1
}

print("========== INDEPENDENT SPLIT AUDIT ==========")
print("Manifest rows:", len(rows))
print("Unique image paths:", len(seen_paths))
print("Identity groups:", len(group_splits))
print("Missing files / other errors:", len(errors))
print("Identity leakage groups:", len(leaking_groups))

print("\nSplit counts:")
for split in ("train", "validation", "test"):
    real = counts[(split, "real")]
    fake = counts[(split, "fake")]
    print(f"{split}: real={real}, fake={fake}, total={real + fake}")

print("\nIdentity groups by split:")
for split in ("train", "validation", "test"):
    groups = sorted(
        group for group, splits in group_splits.items()
        if splits == {split}
    )
    print(f"{split}: {len(groups)} groups -> {groups}")

if errors:
    print("\nFirst 20 errors:")
    for error in errors[:20]:
        print("ERROR:", error)

passed = (
    len(rows) == 10055
    and len(seen_paths) == len(rows)
    and not errors
    and not leaking_groups
    and len(group_splits) == 10
)

print("\nAUDIT PASSED:", passed)
if not passed:
    raise SystemExit(1)
