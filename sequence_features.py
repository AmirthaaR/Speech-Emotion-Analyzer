import librosa
import numpy as np
import pandas as pd
from pathlib import Path


DATASET_PATH = Path("data/RAVDESS/Audio_Speech_Actors_01-24")
OUTPUT_PATH = Path("data/sequence_features.npz")


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


def extract_sequence(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    # Extract MFCC while preserving the time dimension
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return mfcc.T


def main():

    metadata = pd.read_csv("data/ravdess_metadata.csv")

    sequences = []
    labels = []
    actors = []

    print(f"Found {len(metadata)} audio files.")
    print("Extracting MFCC sequences...\n")

    for i, row in metadata.iterrows():

        features = extract_sequence(row["file_path"])

        sequences.append(features)
        labels.append(row["emotion"])
        actors.append(row["actor"])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(metadata)} files")

    # Find the maximum sequence length
    max_length = max(sequence.shape[0] for sequence in sequences)

    print(f"\nMaximum sequence length: {max_length}")
    print(f"MFCC features per frame: {sequences[0].shape[1]}")

    # Pad all sequences to the same length
    X = np.zeros(
        (len(sequences), max_length, 40),
        dtype=np.float32
    )

    for i, sequence in enumerate(sequences):
        length = min(sequence.shape[0], max_length)
        X[i, :length, :] = sequence[:length]

    # Encode labels
    label_map = {
        emotion: index
        for index, emotion in enumerate(EMOTION_MAP.values())
    }

    y = np.array([label_map[label] for label in labels])

    actors = np.array(actors)

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        actors=actors
    )

    print("\nSequence feature extraction completed!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()