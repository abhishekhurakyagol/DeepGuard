import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_v1.pth"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)

model.eval()

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

with torch.no_grad():

    output = model(image)

    probability = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probability, 1)
    classes = ["fake", "real"]

print("="*40)
print("Prediction :", classes[prediction.item()].upper())
print(f"Confidence : {confidence.item()*100:.2f}%")
print("="*40)