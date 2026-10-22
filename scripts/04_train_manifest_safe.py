import csv
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset" / "splits_v3" / "manifest.csv"
MODEL = ROOT / "models" / "deepguard_v1.pth"
CANDIDATE = ROOT / "models" / "deepguard_v1_manifest_candidate.pth"
REPORT_DIR = ROOT / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {"fake": 0, "real": 1}

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest missing: {MANIFEST}")
if not MODEL.exists():
    raise FileNotFoundError(f"Model missing: {MODEL}")

with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

required = {"image_path", "label", "split", "identity_group"}
if not rows or not required.issubset(rows[0]):
    raise ValueError("Manifest is empty or missing required columns.")

groups = defaultdict(set)
for row in rows:
    if row["label"] not in LABELS:
        raise ValueError(f"Unexpected label: {row['label']}")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split: {row['split']}")
    row["_path"] = ROOT / row["image_path"].replace("\\", "/")
    if not row["_path"].is_file():
        raise FileNotFoundError(f"Missing image: {row['_path']}")
    groups[row["identity_group"]].add(row["split"])

leaks = {g: s for g, s in groups.items() if len(s) > 1}
if leaks:
    raise ValueError(f"Identity leakage found: {leaks}")

class ManifestDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        with Image.open(row["_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, LABELS[row["label"]]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

parts = {
    name: [r for r in rows if r["split"] == name]
    for name in ("train", "validation", "test")
}

for name, items in parts.items():
    if not items:
        raise ValueError(f"No images in {name} split.")
    print(f"{name}: {len(items)} images")

train_ds = ManifestDataset(parts["train"], train_tf)
val_ds = ManifestDataset(parts["validation"], eval_tf)
test_ds = ManifestDataset(parts["test"], eval_tf)

workers = 0
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=workers, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=workers, pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=workers, pin_memory=torch.cuda.is_available()
)

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

try:
    state = torch.load(MODEL, map_location="cpu", weights_only=True)
except TypeError:
    state = torch.load(MODEL, map_location="cpu")

if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

model.load_state_dict(state, strict=True)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total

initial_loss, initial_acc = evaluate(val_loader)
best_acc = initial_acc
best_state = {
    k: v.detach().cpu().clone()
    for k, v in model.state_dict().items()
}
history = []

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Initial validation accuracy: {initial_acc:.2f}%")

for epoch in range(EPOCHS):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = loss_sum / total
    train_acc = 100.0 * correct / total
    val_loss, val_acc = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
        f"train_acc={train_acc:.2f}%, val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.2f}%"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
    })

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        torch.save(best_state, CANDIDATE)
        print("New best candidate saved.")

# Always write the candidate separately; never replace the original.
torch.save(best_state, CANDIDATE)
model.load_state_dict(best_state, strict=True)
test_loss, test_acc = evaluate(test_loader)

report = {
    "manifest": str(MANIFEST),
    "total_manifest_images": len(rows),
    "split_counts": {k: len(v) for k, v in parts.items()},
    "initial_validation_accuracy": initial_acc,
    "best_validation_accuracy": best_acc,
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "epochs_completed": len(history),
    "candidate_model": str(CANDIDATE),
    "original_model_modified": False,
    "history": history,
}

report_path = REPORT_DIR / "manifest_finetune_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nTRAINING RUN FINISHED")
print(f"Best validation accuracy: {best_acc:.2f}%")
print(f"Held-out test accuracy: {test_acc:.2f}%")
print(f"Candidate: {CANDIDATE}")
print(f"Report: {report_path}")
print("Original model and backups were not overwritten.")
