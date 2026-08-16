import cv2
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# Project Paths
# ==========================================================

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

REAL_OUTPUT = DATASET_DIR / "extracted_frames" / "real"
FAKE_OUTPUT = DATASET_DIR / "extracted_frames" / "fake"

REAL_OUTPUT.mkdir(parents=True, exist_ok=True)
FAKE_OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Settings
# ==========================================================

FRAME_INTERVAL = 30

# ==========================================================
# Frame Extraction Function
# ==========================================================


def extract_frames(video_path, output_folder, frame_interval=FRAME_INTERVAL):
    """Extract frames from a video."""

    # Skip already processed videos
    existing = list(output_folder.glob(f"{video_path.stem}_*.jpg"))
    if existing:
        print(f"Skipping {video_path.name} (already processed)")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:

            output_file = (
                output_folder
                / f"{video_path.stem}_{saved_count:06d}.jpg"
            )

            cv2.imwrite(str(output_file), frame)

            saved_count += 1

        frame_count += 1

    cap.release()


# ==========================================================
# Process Real Videos
# ==========================================================

print("\n========================================")
print("Extracting REAL videos...")
print("========================================\n")

real_videos = list(REAL_VIDEO_DIR.glob("*.mp4"))

for video in tqdm(real_videos):
    extract_frames(video, REAL_OUTPUT)

# ==========================================================
# Process Fake Videos
# ==========================================================

print("\n========================================")
print("Extracting FAKE videos...")
print("========================================\n")

for folder in FAKE_VIDEO_DIRS:

    print(f"\nProcessing: {folder.parent.parent.name}")

    fake_videos = list(folder.glob("*.mp4"))

    for video in tqdm(fake_videos):
        extract_frames(video, FAKE_OUTPUT)

# ==========================================================
# Finished
# ==========================================================

print("\n========================================")
print("Frame Extraction Completed Successfully!")
print(f"Real Frames : {REAL_OUTPUT}")
print(f"Fake Frames : {FAKE_OUTPUT}")
print("========================================")