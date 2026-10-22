
import cv2
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

REAL_VIDEO_DIR = (
    DATASET_DIR / "original_sequences" / "youtube" / "c23" / "videos"
)

FAKE_VIDEO_DIRS = [
    DATASET_DIR / "manipulated_sequences" / "Deepfakes" / "c23" / "videos",
    DATASET_DIR / "manipulated_sequences" / "Face2Face" / "c23" / "videos",
    DATASET_DIR / "manipulated_sequences" / "FaceSwap" / "c23" / "videos",
    DATASET_DIR / "manipulated_sequences" / "NeuralTextures" / "c23" / "videos",
]

REAL_OUTPUT = DATASET_DIR / "extracted_frames_v2" / "real"
FAKE_OUTPUT = DATASET_DIR / "extracted_frames_v2" / "fake"

# Extract one frame every 5 source frames.
FRAME_INTERVAL = 5
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

def extract_frames(video_path, output_folder, prefix):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return 0

    frame_index = 0
    saved_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % FRAME_INTERVAL == 0:
                output_path = (
                    output_folder
                    / f"{prefix}__{saved_count:06d}.jpg"
                )

                if cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    saved_count += 1
                else:
                    print(f"WARNING: Failed to save {output_path}")

            frame_index += 1
    finally:
        cap.release()

    return saved_count

def find_videos(folder):
    if not folder.exists():
        print(f"WARNING: Missing folder: {folder}")
        return []

    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

def main():
    REAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    FAKE_OUTPUT.mkdir(parents=True, exist_ok=True)

    real_total = 0
    fake_total = 0

    real_videos = find_videos(REAL_VIDEO_DIR)
    print(f"Real source videos: {len(real_videos)}")

    for video in tqdm(real_videos, desc="Extracting real"):
        prefix = f"real__{video.stem}"
        count = extract_frames(video, REAL_OUTPUT, prefix)
        real_total += count
        print(f"{video.name}: {count} frames")

    for folder in FAKE_VIDEO_DIRS:
        method = folder.parent.parent.name
        videos = find_videos(folder)

        print(f"\n{method}: {len(videos)} source videos")

        for video in tqdm(videos, desc=f"Extracting {method}"):
            # Include the method to avoid filename collisions.
            prefix = f"fake__{method}__{video.stem}"
            count = extract_frames(video, FAKE_OUTPUT, prefix)
            fake_total += count

    print("\n========== EXTRACTION COMPLETE ==========")
    print(f"New real frames extracted: {real_total}")
    print(f"New fake frames extracted: {fake_total}")
    print(f"Real output: {REAL_OUTPUT}")
    print(f"Fake output: {FAKE_OUTPUT}")

if __name__ == "__main__":
    main()
