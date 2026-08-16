from backend.video_utils import extract_frames

frames = extract_frames("sample.mp4")

print("=" * 40)
print("Frames Extracted :", len(frames))
print("=" * 40)