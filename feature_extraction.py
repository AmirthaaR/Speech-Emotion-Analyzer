import librosa
import numpy as np
import pandas as pd
from pathlib import Path


DATASET_PATH = Path("data/RAVDESS/Audio_Speech_Actors_01-24")
OUTPUT_PATH = Path("data/features.csv")


EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std])


def main():

    records = []

    wav_files = list(DATASET_PATH.rglob("*.wav"))

    print(f"Found {len(wav_files)} audio files.")
    print("Extracting features...\n")

    for i, wav_file in enumerate(wav_files, start=1):

        parts = wav_file.stem.split("-")

        emotion_code = int(parts[2])
        actor_id = int(parts[6])

        features = extract_features(wav_file)

        record = {
            "file_path": str(wav_file),
            "actor": actor_id,
            "emotion": EMOTION_MAP[emotion_code]
        }

        for j, value in enumerate(features):
            record[f"feature_{j + 1}"] = value

        records.append(record)

        if i % 100 == 0:
            print(f"Processed {i}/{len(wav_files)} files")

    df = pd.DataFrame(records)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction completed!")
    print(f"Dataset shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()