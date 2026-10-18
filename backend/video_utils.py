import cv2
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_frames(video_path, frame_interval=15):
    """
    Extract frames from a video.
    Maintains backward compatibility with original DeepGuard signature.

    Args:
        video_path (str or Path): Path to video
        frame_interval (int): Extract one frame every N frames

    Returns:
        list: List of extracted frames (OpenCV BGR images)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

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


def extract_frames_with_metadata(
    video_path,
    max_frames: int = 60,
    frame_interval: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract video frames with timestamp metadata, limits, and video duration safeguards.

    Returns:
        (frames_data, video_metadata)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = round(total_video_frames / fps, 2) if total_video_frames > 0 else 0.0

    # Dynamically adjust step if video has more frames than max_frames allows
    step = frame_interval
    if total_video_frames > 0 and (total_video_frames / frame_interval) > max_frames:
        step = max(frame_interval, int(total_video_frames / max_frames))

    frames_data = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % step == 0:
            timestamp = round(frame_count / fps, 2)
            frames_data.append({
                "frame": frame,
                "frame_index": frame_count,
                "timestamp_sec": timestamp,
            })
            if len(frames_data) >= max_frames:
                break

        frame_count += 1

    cap.release()

    meta = {
        "fps": round(fps, 2),
        "total_video_frames": total_video_frames,
        "duration_sec": duration_sec,
        "sampled_frames_count": len(frames_data),
        "step_interval": step,
    }

    return frames_data, meta


def save_uploaded_video(upload_file):
    """
    Save FastAPI uploaded video to a temporary file.

    Returns:
        Path to saved video.
    """
    suffix = Path(upload_file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )
    temp_file.write(upload_file.file.read())
    temp_file.close()
    return temp_file.name