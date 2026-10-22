
from pathlib import Path
import shutil
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm

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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "deepguard_v1.pth"
BACKUP_PATH = MODEL_DIR / "deepguard_v1_backup.pth"
STAGED_MODEL_PATH = MODEL_DIR / "deepguard_v1_finetuned.pth"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("DeepGuard: Fine-tuning existing model")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)

# ==========================================================
# Validate inputs
# ==========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Existing model not found: {MODEL_PATH}")

if not BACKUP_PATH.exists():
    raise FileNotFoundError(
        f"Backup not found: {BACKUP_PATH}. Back up the existing model first."
    )

for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

# ==========================================================
# Transforms
# ==========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ==========================================================
# Dataset and class validation
# ==========================================================

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

EXPECTED_CLASSES = ["fake", "real"]

for name, dataset in (
    ("train", train_dataset),
    ("validation", val_dataset),
    ("test", test_dataset),
):
    if dataset.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"{name} class order is {dataset.classes}; "
            f"expected {EXPECTED_CLASSES}. Check folder names."
        )
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty.")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))
print("Classes:", train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# ==========================================================
# Load EXISTING DeepGuard weights
# ==========================================================

print("\nLoading existing DeepGuard model...")

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES,
)

# This file should contain the plain model state_dict.
try:
    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    # Compatibility with older PyTorch versions.
    state = torch.load(MODEL_PATH, map_location="cpu")

if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

model.load_state_dict(state, strict=True)
model = model.to(DEVICE)
print("Existing weights loaded successfully.")

# ==========================================================
# Fine-tuning setup
# ==========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# Start with the current model as the best candidate.
# This prevents a worse fine-tuned model from replacing it.
best_accuracy = None
best_state = None

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

# Measure the current model before fine-tuning.
initial_val_loss, initial_val_acc = evaluate(val_loader)
best_accuracy = initial_val_acc
best_state = {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
}

print(f"Initial validation accuracy: {initial_val_acc:.2f}%")

# ==========================================================
# Fine-tuning loop
# ==========================================================

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    val_loss, val_acc = evaluate(val_loader)
    scheduler.step(val_acc)

    print(f"\nTrain loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_acc:.2f}%")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.2f}%")

    # Retain only a model that improves on the original baseline.
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        print("Improved model candidate recorded.")

# ==========================================================
# Save candidate without overwriting the existing model
# ==========================================================

torch.save(best_state, STAGED_MODEL_PATH)

# Verify that the candidate can be loaded.
verification_model = efficientnet_b0(weights=None)
verification_model.classifier[1] = nn.Linear(
    verification_model.classifier[1].in_features,
    NUM_CLASSES,
)
verification_model.load_state_dict(
    torch.load(
        STAGED_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    ),
    strict=True,
)

# Test the candidate on the test split.
model.load_state_dict(best_state, strict=True)
test_loss, test_acc = evaluate(test_loader)

print("\n" + "=" * 60)
print(f"Original validation accuracy: {initial_val_acc:.2f}%")
print(f"Best candidate validation accuracy: {best_accuracy:.2f}%")
print(f"Candidate test accuracy: {test_acc:.2f}%")
print("Candidate saved to:", STAGED_MODEL_PATH)
print("=" * 60)

if best_accuracy > initial_val_acc:
    # Preserve the current file until the candidate has been validated.
    shutil.copy2(STAGED_MODEL_PATH, MODEL_PATH)
    print("Existing DeepGuard model updated successfully.")
else:
    print("No validation improvement; existing model was NOT replaced.")
    print("Review the dataset and training results before trying again.")

torch.save(
    {
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_accuracy": best_accuracy,
        "initial_validation_accuracy": initial_val_acc,
        "test_accuracy": test_acc,
    },
    LATEST_CHECKPOINT,
)

print("Training run completed.")
