import csv
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
STAGING_ROOT = PROJECT_ROOT / "dataset_expansion_staging"
OUTPUT_ROOT = DATASET_ROOT / "extracted_frames_v3_complete"

FRAME_INTERVAL = 5
JPEG_QUALITY = 95
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

REAL_PARTS = ("original_sequences", "youtube", "c23", "videos")
FAKE_PREFIX = ("manipulated_sequences",)
METHODS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}


def relative_key(path, root):
    return str(path.relative_to(root)).replace("\\", "/").lower()


def identity_group(stem, is_real):
    if is_real:
        return stem

    # Fake filenames such as 033_097 and 097_033
    ids = stem.split("_")
    if len(ids) == 2 and all(ids):
        return "_".join(sorted(ids))

    return f"UNKNOWN_{stem}"


def classify(relative):
    parts = relative.replace(chr(92), "/").split("/")

    if (
        len(parts) == 5
        and parts[0] == "original_sequences"
        and parts[1:4] == ["youtube", "c23", "videos"]
    ):
        return {"label": "real", "method": "original"}

    if (
        len(parts) == 5
        and parts[0] == "manipulated_sequences"
        and parts[1] in METHODS
        and parts[2:4] == ["c23", "videos"]
    ):
        return {"label": "fake", "method": parts[1]}

    return None


def main():
    if not STAGING_ROOT.exists():
        raise SystemExit(f"Staging folder missing: {STAGING_ROOT}")

    if OUTPUT_ROOT.exists():
        raise SystemExit(
            f"STOP: output already exists: {OUTPUT_ROOT}\n"
            "Inspect it first. This script will not overwrite or mix outputs."
        )

    # Compare relative paths to exclude files already present in dataset.
    existing = set()
    if DATASET_ROOT.exists():
        for p in DATASET_ROOT.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".mp4":
                existing.add(relative_key(p, DATASET_ROOT))

    candidates = []
    for p in STAGING_ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        rel = p.relative_to(STAGING_ROOT)
        key = str(rel).replace("\\", "/").lower()

        if key in existing:
            continue

        info = classify(str(rel))
        if info is None:
            continue

        candidates.append((p, str(rel), info))

    candidates.sort(key=lambda item: item[1].lower())

    if len(candidates) != 50:
        raise SystemExit(
            f"STOP: expected exactly 50 new supported videos, found "
            f"{len(candidates)}. No output was created."
        )

    real_count = sum(info["label"] == "real" for _, _, info in candidates)
    fake_count = sum(info["label"] == "fake" for _, _, info in candidates)

    method_counts = {
        method: sum(
            info["method"] == method for _, _, info in candidates
        )
        for method in sorted(METHODS)
    }

    if real_count != 10 or fake_count != 40 or any(
        count != 10 for count in method_counts.values()
    ):
        raise SystemExit(
            "STOP: unexpected category counts. No output was created.\n"
            f"Real={real_count}, Fake={fake_count}, methods={method_counts}"
        )

    # Only create the new output directory after preflight passes.
    (OUTPUT_ROOT / "real").mkdir(parents=True)
    (OUTPUT_ROOT / "fake").mkdir(parents=True)

    manifest = OUTPUT_ROOT / "extraction_manifest.csv"
    rows = []

    for video, relative, info in candidates:
        is_real = info["label"] == "real"
        label = info["label"]
        method = info["method"]
        group = identity_group(video.stem, is_real)

        output_dir = OUTPUT_ROOT / label
        prefix = f"{label}__{method}__{video.stem}"

        cap = cv2.VideoCapture(str(video))
        frames_read = 0
        frames_saved = 0
        failed = not cap.isOpened()

        try:
            if not failed:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frames_read % FRAME_INTERVAL == 0:
                        out = output_dir / f"{prefix}__{frames_read:06d}.jpg"
                        if cv2.imwrite(
                            str(out), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                        ):
                            frames_saved += 1
                        else:
                            failed = True
                            print("SAVE FAILED:", out)

                    frames_read += 1
        finally:
            cap.release()

        status = (
            "PASS" if not failed and frames_read > 0 and frames_saved > 0
            else "FAIL"
        )

        rows.append({
            "source_video": relative,
            "label": label,
            "method": method,
            "identity_group": group,
            "frames_read": frames_read,
            "frames_saved": frames_saved,
            "status": status,
        })

        print(
            f"{status}: {relative} | decoded={frames_read} "
            f"| saved={frames_saved}"
        )

    fields = [
        "source_video", "label", "method", "identity_group",
        "frames_read", "frames_saved", "status"
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Videos processed:", len(rows))
    print("Videos passed:", sum(r["status"] == "PASS" for r in rows))
    print("Videos failed:", sum(r["status"] == "FAIL" for r in rows))
    print("Real frames:", sum(r["frames_saved"] for r in rows if r["label"] == "real"))
    print("Fake frames:", sum(r["frames_saved"] for r in rows if r["label"] == "fake"))
    print("Output:", OUTPUT_ROOT)
    print("Manifest:", manifest)
    print("Training has NOT been started.")


if __name__ == "__main__":
    main()
