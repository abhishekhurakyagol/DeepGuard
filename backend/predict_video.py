from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from video_utils import extract_frames

# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_v1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Load Model
# ==========================================

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

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
        std=[0.229, 0.224, 0.225]
    )
])


def predict_video(video_path):

    frames = extract_frames(video_path)

    fake_count = 0
    real_count = 0
    confidences = []

    for frame in frames:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)

        image = transform(image)
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        label = classes[prediction.item()]
        confidences.append(confidence.item())

        if label == "fake":
            fake_count += 1
        else:
            real_count += 1

    total = fake_count + real_count

    final_prediction = "fake" if fake_count > real_count else "real"

    avg_confidence = (
        sum(confidences) / len(confidences) * 100
        if confidences
        else 0
    )

    return {
        "prediction": final_prediction,
        "confidence": round(avg_confidence, 2),
        "fake_frames": fake_count,
        "real_frames": real_count,
        "total_frames": total
    }