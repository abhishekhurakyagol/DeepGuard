
from pathlib import Path
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0
from PIL import ImageFile
from PIL import Image

from audio_utils import audio_to_melspectrogram

ImageFile.LOAD_TRUNCATED_IMAGES = False

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_ROOT = PROJECT_ROOT / "datasets" / "audio"
CACHE_ROOT = PROJECT_ROOT / "datasets" / "audio_mels"
MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_audio.pth"

# Keep these settings identical to predict_audio.py
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def prepare_spectrograms():
    """Convert audio files into cached mel-spectrogram images."""
    total = 0

    for label in ("fake", "real"):
        source_dir = AUDIO_ROOT / label
        output_dir = CACHE_ROOT / label

        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {source_dir}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = [
            p for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ]

        if not audio_files:
            raise ValueError(
                f"No supported audio files found in {source_dir}"
            )

        for index, audio_path in enumerate(audio_files, start=1):
            # Include the relative path in the cache filename.
            relative = audio_path.relative_to(source_dir)
            cache_name = "__".join(relative.with_suffix("").parts)
            image_path = output_dir / f"{cache_name}.png"

            if image_path.exists():
                total += 1
                continue

            temp_image = None

            try:
                temp_image = audio_to_melspectrogram(str(audio_path))
                with Image.open(temp_image) as image:
                    image.convert("RGB").save(image_path, format="PNG")

                total += 1
                print(f"[{label}] {index}/{len(audio_files)}: {audio_path.name}")

            except Exception as exc:
                if image_path.exists():
                    image_path.unlink()
                print(f"Skipping unreadable file {audio_path}: {exc}")

            finally:
                if temp_image:
                    try:
                        Path(temp_image).unlink(missing_ok=True)
                    except OSError:
                        pass

    if total == 0:
        raise RuntimeError("No audio spectrograms were prepared.")

    print(f"Prepared or reused {total} cached spectrograms.")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Training device: {DEVICE}")

    prepare_spectrograms()

    dataset = ImageFolder(
        root=str(CACHE_ROOT),
        transform=transform,
    )

    # ImageFolder sorts alphabetically: fake=0, real=1.
    if dataset.class_to_idx != {"fake": 0, "real": 1}:
        raise RuntimeError(
            f"Unexpected class mapping: {dataset.class_to_idx}. "
            "Expected fake=0 and real=1."
        )

    fake_count = sum(label == 0 for _, label in dataset.samples)
    real_count = sum(label == 1 for _, label in dataset.samples)

    print(f"Fake samples: {fake_count}")
    print(f"Real samples: {real_count}")

    if fake_count < 2 or real_count < 2:
        raise ValueError(
            "Add at least two usable audio files to each class. "
            "More diverse samples are strongly recommended."
        )

    validation_size = max(1, round(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size

    if training_size < 2:
        raise ValueError("Not enough audio samples for training and validation.")

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2,
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    best_val_accuracy = -1.0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_accuracy = 100 * train_correct / max(1, train_total)
        val_accuracy = 100 * val_correct / max(1, val_total)
        average_loss = train_loss / max(1, train_total)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2f}% | "
            f"Validation accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best checkpoint: {MODEL_PATH}")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Model checkpoint: {MODEL_PATH}")


if __name__ == "__main__":
    main()
