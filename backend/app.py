from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException

from torchvision import transforms
from torchvision.models import efficientnet_b0
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_v1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="DeepGuard API",
    version="1.0.0",
    description="AI Deepfake Detection API"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
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

print("===================================")
print(" DeepGuard Model Loaded Successfully")
print(" Device :", DEVICE)
print("===================================")

classes = [
    "fake",
    "real"
]

@app.get("/")
def home():
    return {
        "message": "DeepGuard API is Running",
        "model": "EfficientNet-B0",
        "status": "Ready"
    }

def predict_image(image: Image.Image):

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    image = Image.open(file.file).convert("RGB")

    result = predict_image(image)

    return result

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
