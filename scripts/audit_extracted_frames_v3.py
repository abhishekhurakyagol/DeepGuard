import csv
import hashlib
from collections import Counter
from pathlib import Path
import cv2

root = Path("dataset/extracted_frames_v3_complete")
manifest_path = root / "extraction_manifest.csv"

if not manifest_path.exists():
    raise SystemExit(f"STOP: missing manifest: {manifest_path}")

with manifest_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("========== MANIFEST AUDIT ==========")
print("Manifest rows:", len(rows))
print("Videos passed:", sum(r["status"] == "PASS" for r in rows))
print("Videos failed:", sum(r["status"] != "PASS" for r in rows))

print("\nFrames saved according to manifest:")
for label in ("real", "fake"):
    total = sum(int(r["frames_saved"]) for r in rows if r["label"] == label)
    print(f"{label}: {total}")

actual = {}
for label in ("real", "fake"):
    folder = root / label
    files = sorted(folder.glob("*.jpg")) if folder.exists() else []
    actual[label] = files
    print(f"Actual {label} JPG files: {len(files)}")

print("\n========== IMAGE VALIDATION ==========")
unreadable = []
hashes = {}
duplicate_count = 0

for label, files in actual.items():
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            unreadable.append(str(path))
            continue

        # Detect exact duplicate image files by content hash.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            duplicate_count += 1
        else:
            hashes[digest] = str(path)

print("Unreadable images:", len(unreadable))
print("Repeated exact-image occurrences:", duplicate_count)

if unreadable:
    print("\nFirst unreadable files:")
    for name in unreadable[:20]:
        print(name)

print("\n========== FINAL CHECK ==========")
manifest_totals = {
    label: sum(int(r["frames_saved"]) for r in rows if r["label"] == label)
    for label in ("real", "fake")
}
counts_match = all(manifest_totals[label] == len(actual[label]) for label in ("real", "fake"))
videos_pass = len(rows) == 50 and all(r["status"] == "PASS" for r in rows)
images_readable = not unreadable

print("Manifest matches actual file counts:", counts_match)
print("All 50 videos passed:", videos_pass)
print("All images readable:", images_readable)
print("AUDIT PASSED:", counts_match and videos_pass and images_readable)
print("Note: exact duplicate detection does not detect near-duplicate frames.")
