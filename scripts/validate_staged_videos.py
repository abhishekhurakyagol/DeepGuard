import cv2
import csv
from pathlib import Path

root = Path("dataset_expansion_staging")
report_path = root / "video_validation.csv"

# Validate only new paths identified by the previous audit.
# Rebuild this list from staging against the existing dataset.
existing_root = Path("dataset")
existing = {
    str(p.relative_to(existing_root)).replace("\\", "/").lower()
    for p in existing_root.rglob("*.mp4")
}

results = []

for path in sorted(root.rglob("*.mp4")):
    if path.name.lower() == "video_inventory.csv":
        continue

    relative = str(path.relative_to(root)).replace("\\", "/")
    key = relative.lower()

    if key in existing:
        continue

    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frame_ok = False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0

    if opened:
        ok, frame = cap.read()
        frame_ok = bool(ok and frame is not None)

    cap.release()

    results.append({
        "path": relative,
        "opened": opened,
        "first_frame_decoded": frame_ok,
        "reported_frames": frame_count,
        "width": width,
        "height": height,
        "status": "PASS" if opened and frame_ok and frame_count > 0 else "FAIL"
    })

with report_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [
        "path", "opened", "first_frame_decoded", "reported_frames",
        "width", "height", "status"
    ])
    writer.writeheader()
    writer.writerows(results)

print("New videos checked:", len(results))
print("PASS:", sum(r["status"] == "PASS" for r in results))
print("FAIL:", sum(r["status"] == "FAIL" for r in results))
print("Report:", report_path)

for r in results:
    if r["status"] == "FAIL":
        print("FAILED:", r["path"], "|", r["status"])
