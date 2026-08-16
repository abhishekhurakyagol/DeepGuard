from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "deepguard_v1.pth"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "deepguard_v1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print("Model Loaded Successfully!")

dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

scripted_model = torch.jit.trace(model, dummy_input)

script_path = OUTPUT_DIR / "deepguard_torchscript.pt"

scripted_model.save(script_path)

print("TorchScript Model Saved!")

classes = ["fake", "real"]

with open(OUTPUT_DIR / "classes.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print("Class Labels Saved!")

info = {
    "model": "EfficientNet-B0",
    "classes": classes,
    "image_size": 224,
    "framework": "PyTorch",
    "author": "Abhishek Hurakyagol"
}

with open(OUTPUT_DIR / "model_info.json", "w") as f:
    json.dump(info, f, indent=4)

print("Model Information Saved!")

print("\n====================================")
print("DeepGuard Export Completed!")
print("Files saved in:", OUTPUT_DIR)
print("====================================")