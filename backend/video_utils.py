import cv2
import tempfile
from pathlib import Path


def extract_frames(video_path, frame_interval=15):
    """
    Extract frames from a video.

    Args:
        video_path (str or Path): Path to video
        frame_interval (int): Extract one frame every N frames

    Returns:
        list: List of extracted frames (OpenCV images)
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    frames = []
    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    return frames


def save_uploaded_video(upload_file):
    """
    Save FastAPI uploaded video to a temporary file.

    Returns:
        Path to saved video.
    """

    suffix = Path(upload_file.filename).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(upload_file.file.read())
    temp_file.close()

    return temp_file.name