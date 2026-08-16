from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = MODEL_DIR / "deepguard_v1.pth"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Configuration
# ==========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

print("=" * 60)

# ==========================================================
# Transform
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==========================================================
# Dataset
# ==========================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Test Images :", len(test_dataset))
print("Classes :", test_dataset.classes)

# ==========================================================
# Load Model
# ==========================================================

model = efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)
model.eval()

print("\nModel Loaded Successfully!")

# ==========================================================
# Evaluation
# ==========================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(all_labels, all_predictions)

print("\n" + "=" * 60)
print("Test Accuracy : {:.2f}%".format(accuracy * 100))
print("=" * 60)

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes,
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(ax=ax)

plt.title("DeepGuard Confusion Matrix")

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nConfusion Matrix saved to:")
print(OUTPUT_DIR / "confusion_matrix.png")

print("\nEvaluation Completed Successfully!")