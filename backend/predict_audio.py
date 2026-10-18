from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

try:
    from audio_utils import process_audio_analysis, audio_to_melspectrogram
except ModuleNotFoundError:
    from backend.audio_utils import process_audio_analysis, audio_to_melspectrogram

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepguard_audio.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["fake", "real"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model = None


def load_audio_model():
    """Load the trained model only when an audio prediction is requested."""
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The audio detection model has not been trained yet. "
            "Model weights deepguard_audio.pth not found in models directory."
        )

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


def predict_audio(audio_path: str) -> Dict[str, Any]:
    """Run real inference on an audio file and extract visual/forensic assets."""
    model = load_audio_model()
    temp_spec_path = None

    try:
        temp_spec_path, b64_spectrogram, waveform, duration_sec, observations, feature_weights = (
            process_audio_analysis(audio_path)
        )

        with Image.open(temp_spec_path) as image:
            image_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = probabilities.max(dim=1)

        pred_label = CLASSES[prediction.item()]
        conf_percent = round(confidence.item() * 100, 2)

        return {
            "prediction": pred_label,
            "confidence": conf_percent,
            "spectrogram": b64_spectrogram,
            "waveform": waveform,
            "duration_sec": duration_sec,
            "media_type": "audio",
            "forensic_observations": observations,
            "feature_weights": feature_weights,
            "transcription": {
                "status": "not_configured",
                "text": None,
                "message": "Speech-to-text whisper model is not currently configured.",
            },
        }

    finally:
        if temp_spec_path:
            try:
                Path(temp_spec_path).unlink(missing_ok=True)
            except OSError:
                pass
