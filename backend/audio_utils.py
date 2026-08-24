from pathlib import Path
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt


# ==========================================
# Convert Audio -> Mel Spectrogram
# ==========================================

def audio_to_melspectrogram(audio_path):

    # Load audio
    signal, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=128,
        fmax=8000
    )

    # Convert to dB
    mel_db = librosa.power_to_db(
        mel,
        ref=max
    )

    # Temporary PNG file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    # Create figure
    plt.figure(figsize=(4, 4))

    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis=None,
        y_axis=None,
        cmap="magma"
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        temp_file.name,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return temp_file.name