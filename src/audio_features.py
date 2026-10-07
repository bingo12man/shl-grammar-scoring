"""Handcrafted acoustic feature extraction used by the multimodal system.

This public version contains only reusable feature-extraction logic and no
competition data paths or protected artifacts.
"""

import numpy as np
import librosa


def extract_audio_features(path: str, sr: int = 16000) -> dict:
    y, _ = librosa.load(path, sr=sr, mono=True)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {
        "duration": len(y) / sr,
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
        "centroid_mean": float(centroid.mean()),
        "centroid_std": float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "bandwidth_std": float(bandwidth.std()),
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_std": float(rolloff.std()),
    }

    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(mfcc[i].mean())
        features[f"mfcc_{i}_std"] = float(mfcc[i].std())

    return features
