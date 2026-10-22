
import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path(__file__).resolve().parent.parent
SOURCES = [
    PROJECT / "dataset" / "extracted_frames_v2",
    PROJECT / "dataset" / "extracted_frames_v3_complete",
]
OUTPUT = PROJECT / "dataset" / "splits_v3"

if OUTPUT.exists():
    raise SystemExit(f"STOP: output already exists: {OUTPUT}")

# Identify source identity pairs from fake-frame filenames.
pairs = set()
files_by_source = []

for root in SOURCES:
    if not root.exists():
        raise SystemExit(f"STOP: missing source directory: {root}")

    for label in ("real", "fake"):
        folder = root / label
        if not folder.exists():
            raise SystemExit(f"STOP: missing label folder: {folder}")

        for path in sorted(folder.glob("*.jpg")):
            parts = path.stem.split("__")

            if label == "fake":
                if len(parts) < 4 or parts[0] != "fake":
                    raise SystemExit(f"STOP: unknown fake filename: {path.name}")
                ids = parts[2].split("_")
                if len(ids) != 2:
                    raise SystemExit(f"STOP: unknown identity pair: {path.name}")
                pairs.add(tuple(sorted(ids)))
            else:
                if len(parts) >= 4 and parts[1] == "original":
                    source_id = parts[2]
                elif len(parts) >= 3:
                    source_id = parts[1]
                else:
                    raise SystemExit(f"STOP: unknown real filename: {path.name}")

            files_by_source.append((root, path, label))

# Union-find ensures connected identities stay in one split.
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

for a, b in pairs:
    union(a, b)

def group_for_pair(stem):
    ids = stem.split("_")
    return find(ids[0])

# Resolve every original identity to its corresponding fake-pair group.
identity_group = {}
for a, b in pairs:
    root = find(a)
    identity_group[a] = root
    identity_group[b] = root

groups = sorted({find(a) for pair in pairs for a in pair})
if len(groups) != 10:
    raise SystemExit(
        f"STOP: expected 10 identity groups, found {len(groups)}. "
        "Inspect the source filenames before proceeding."
    )

# Deterministic split assignment at group level.
rng = random.Random(42)
shuffled = groups.copy()
rng.shuffle(shuffled)

split_for_group = {}
for i, group in enumerate(shuffled):
    if i < 7:
        split_for_group[group] = "train"
    elif i == 7:
        split_for_group[group] = "validation"
    else:
        split_for_group[group] = "test"

rows = []
for root, path, label in files_by_source:
    parts = path.stem.split("__")

    if label == "fake":
        pair_ids = parts[2].split("_")
        group = find(pair_ids[0])
        method = parts[1]
    else:
        if len(parts) >= 4 and parts[1] == "original":
            source_id = parts[2]
        else:
            source_id = parts[1]

        if source_id not in identity_group:
            raise SystemExit(
                f"STOP: real identity {source_id} has no fake-pair group."
            )
        group = identity_group[source_id]
        method = "original"

    rows.append({
        "image_path": str(path.relative_to(PROJECT)),
        "label": label,
        "method": method,
        "identity_group": group,
        "split": split_for_group[group],
        "source_version": root.name,
    })

# Validate group isolation and label totals before writing.
group_splits = defaultdict(set)
for row in rows:
    group_splits[row["identity_group"]].add(row["split"])

if any(len(splits) != 1 for splits in group_splits.values()):
    raise SystemExit("STOP: identity leakage detected.")

if len(rows) != 10055:
    raise SystemExit(
        f"STOP: expected 10055 frames, found {len(rows)}. "
        "No manifest was written."
    )

OUTPUT.mkdir(parents=True)
with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Manifest created:", OUTPUT / "manifest.csv")
print("Total frames:", len(rows))
print("Groups:", len(group_splits))
print("Frames by split and label:")

counts = Counter((row["split"], row["label"]) for row in rows)
for split in ("train", "validation", "test"):
    print(
        f"{split}: real={counts[(split, 'real')]}, "
        f"fake={counts[(split, 'fake')]}"
    )

print("Identity leakage:", "NONE" if all(
    len(splits) == 1 for splits in group_splits.values()
) else "DETECTED")
print("Training has NOT been started.")
