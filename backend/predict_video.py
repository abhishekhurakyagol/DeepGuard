from pathlib import Path
from typing import Dict, Any, List
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

try:
    from video_utils import extract_frames_with_metadata, extract_frames
except ModuleNotFoundError:
    from backend.video_utils import extract_frames_with_metadata, extract_frames

try:
    from gradcam import generate_gradcam_overlay
except ModuleNotFoundError:
    from backend.gradcam import generate_gradcam_overlay

# ==========================================
# Paths & Settings
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_v1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Load Model
# ==========================================
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

classes = ["fake", "real"]

# ==========================================
# Transform
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_video(video_path: str, max_frames: int = 60, frame_interval: int = 15) -> Dict[str, Any]:
    """
    Perform deepfake inference on extracted video frames with timestamps and timeline analysis.
    """
    frames_data, meta = extract_frames_with_metadata(
        video_path,
        max_frames=max_frames,
        frame_interval=frame_interval,
    )

    if not frames_data:
        raise ValueError("No video frames could be extracted from the provided file.")

    fake_count = 0
    real_count = 0
    confidences: List[float] = []
    timeline: List[Dict[str, Any]] = []

    highest_fake_conf = -1.0
    top_fake_image = None
    top_fake_timestamp = None

    for item in frames_data:
        frame_bgr = item["frame"]
        timestamp = item["timestamp_sec"]
        frame_idx = item["frame_index"]

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        tensor_img = transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor_img)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        label = classes[prediction.item()]
        conf_val = round(confidence.item() * 100, 2)
        confidences.append(confidence.item())

        timeline.append({
            "frame_index": frame_idx,
            "timestamp_sec": timestamp,
            "prediction": label,
            "confidence": conf_val,
        })

        if label == "fake":
            fake_count += 1
            if confidence.item() > highest_fake_conf:
                highest_fake_conf = confidence.item()
                top_fake_image = pil_image
                top_fake_timestamp = timestamp
        else:
            real_count += 1

    total = fake_count + real_count
    final_prediction = "fake" if fake_count > real_count else "real"
    avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0.0

    # Generate Grad-CAM evidence for top fake frame if fake was found, else first frame
    evidence_heatmap = None
    forensic_observations = []
    feature_weights = []

    try:
        sample_img = top_fake_image if top_fake_image else Image.fromarray(cv2.cvtColor(frames_data[0]["frame"], cv2.COLOR_BGR2RGB))
        sample_class_idx = 0 if final_prediction == "fake" else 1
        evidence_heatmap, forensic_observations, feature_weights = generate_gradcam_overlay(
            model,
            sample_img,
            transform,
            DEVICE,
            sample_class_idx,
        )
    except Exception:
        pass

    return {
        "prediction": final_prediction,
        "confidence": round(avg_confidence, 2),
        "fake_frames": fake_count,
        "real_frames": real_count,
        "total_frames": total,
        "duration_sec": meta["duration_sec"],
        "fps": meta["fps"],
        "media_type": "video",
        "timeline": timeline,
        "heatmap": evidence_heatmap,
        "forensic_observations": forensic_observations,
        "feature_weights": feature_weights,
        "top_fake_timestamp": top_fake_timestamp,
    }