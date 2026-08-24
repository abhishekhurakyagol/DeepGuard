from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

from audio_utils import audio_to_melspectrogram

# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_audio.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Load Model
# ==========================================

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

classes = [
    "fake",
    "real"
]

# ==========================================
# Image Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Audio Prediction
# ==========================================

def predict_audio(audio_path):

    spectrogram = audio_to_melspectrogram(audio_path)

    image = Image.open(spectrogram).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    return {
        "prediction": classes[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }