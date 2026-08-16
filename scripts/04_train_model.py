import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from torch.utils.tensorboard import SummaryWriter
# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Training Parameters
# -----------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 1e-4

NUM_CLASSES = 2

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

print("=" * 60)

# ==========================================================
# Image Transforms
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

val_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ==========================================================
# Dataset
# ==========================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_transform
)

print()

print("Train Images :", len(train_dataset))

print("Validation Images :", len(val_dataset))

print("Test Images :", len(test_dataset))

print()

print("Classes :", train_dataset.classes)

print()

# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)

print("="*60)

print("DataLoader Ready")

print("="*60)


# ==========================================================
# Load Pretrained EfficientNet-B0
# ==========================================================

print("\nLoading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

# Replace final classifier
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print("Model Loaded Successfully!")

# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.CrossEntropyLoss()

# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2

)

# ==========================================================
# Mixed Precision (RTX 3050)
# ==========================================================

scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)

print("Optimizer Ready")
print("Scheduler Ready")
print("Mixed Precision Enabled:", torch.cuda.is_available())

# ==========================================================
# Statistics
# ==========================================================

best_accuracy = 0.0

train_loss_history = []
val_loss_history = []

train_acc_history = []
val_acc_history = []

print("=" * 60)
print("Training Pipeline Ready")
print("=" * 60)

# ==========================================================
# Training Loop
# ==========================================================

from tqdm import tqdm

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 60)

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in train_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100*correct/total:.2f}%"
        )

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        val_bar = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in val_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            val_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%"
            )

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    scheduler.step(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")

    # -------------------------
    # Save Best Model
    # -------------------------

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            model.state_dict(),
            MODEL_DIR / "deepguard_v1.pth"
        )

        print("Best Model Saved!")

    # -------------------------
    # Save Latest Checkpoint
    # -------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
        },
        CHECKPOINT_DIR / "latest_model.pth"
    )

print("\nTraining Completed Successfully!")
print(f"Best Validation Accuracy : {best_accuracy:.2f}%")