import librosa
import numpy as np
import pandas as pd
from pathlib import Path


DATASET_PATH = Path(
    "data/RAVDESS/Audio_Speech_Actors_01-24"
)

OUTPUT_PATH = Path(
    "data/improved_sequence_features.npz"
)


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

    audio, sample_rate = librosa.load(
        file_path,
        sr=22050
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # First-order temporal derivative
    delta = librosa.feature.delta(mfcc)

    # Second-order temporal derivative
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine all features
    combined = np.concatenate(
        [mfcc, delta, delta_delta],
        axis=0
    )

    # Convert:
    # 120 features × time
    # into:
    # time × 120 features
    return combined.T


def main():

    metadata = pd.read_csv(
        "data/ravdess_metadata.csv"
    )

    sequences = []
    labels = []
    actors = []
    lengths = []

    print(f"Found {len(metadata)} audio files.")
    print("Extracting improved features...\n")

    for i, row in metadata.iterrows():

        features = extract_features(
            row["file_path"]
        )

        sequences.append(features)
        labels.append(row["emotion"])
        actors.append(row["actor"])
        lengths.append(features.shape[0])

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(metadata)} files"
            )

    max_length = max(lengths)

    print(f"\nMaximum sequence length: {max_length}")
    print("Features per frame: 120")

    # Create padded dataset
    X = np.zeros(
        (
            len(sequences),
            max_length,
            120
        ),
        dtype=np.float32
    )

    for i, sequence in enumerate(sequences):

        length = min(
            sequence.shape[0],
            max_length
        )

        X[i, :length, :] = sequence[:length]

    # Label encoding
    label_map = {
        emotion: index
        for index, emotion in enumerate(
            EMOTION_MAP.values()
        )
    }

    y = np.array([
        label_map[label]
        for label in labels
    ])

    actors = np.array(actors)
    lengths = np.array(lengths)

    # Save everything
    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        actors=actors,
        lengths=lengths
    )

    print("\n================================")
    print("IMPROVED FEATURE EXTRACTION")
    print("================================")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Saved to:", OUTPUT_PATH)
    print("================================")


if __name__ == "__main__":
    main()