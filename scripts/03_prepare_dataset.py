import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

SOURCE_REAL = DATASET_DIR / "extracted_frames" / "real"
SOURCE_FAKE = DATASET_DIR / "extracted_frames" / "fake"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ==========================================================
# Create Output Folders
# ==========================================================

OUTPUT_DIRS = [
    DATASET_DIR / "train" / "real",
    DATASET_DIR / "train" / "fake",
    DATASET_DIR / "validation" / "real",
    DATASET_DIR / "validation" / "fake",
    DATASET_DIR / "test" / "real",
    DATASET_DIR / "test" / "fake",
]

for folder in OUTPUT_DIRS:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Split Function
# ==========================================================

def prepare_class(source_dir, class_name):

    images = list(source_dir.glob("*.jpg"))

    random.shuffle(images)

    train_images, temp_images = train_test_split(
        images,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.5,
        random_state=RANDOM_SEED,
    )

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images,
    }

    for split_name, image_list in splits.items():

        destination = DATASET_DIR / split_name / class_name

        for image in image_list:
            shutil.copy2(image, destination)

    return len(train_images), len(val_images), len(test_images)


# ==========================================================
# Main
# ==========================================================

print("\nPreparing Dataset...\n")

real_train, real_val, real_test = prepare_class(
    SOURCE_REAL,
    "real",
)

fake_train, fake_val, fake_test = prepare_class(
    SOURCE_FAKE,
    "fake",
)

print("\n===============================")

print("REAL")
print(f"Train      : {real_train}")
print(f"Validation : {real_val}")
print(f"Test       : {real_test}")

print("\nFAKE")
print(f"Train      : {fake_train}")
print(f"Validation : {fake_val}")
print(f"Test       : {fake_test}")

print("\nDataset preparation completed!")

print("===============================")