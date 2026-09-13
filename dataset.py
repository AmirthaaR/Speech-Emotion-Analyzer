from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/RAVDESS/Audio_Speech_Actors_01-24")

emotion_map = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

records = []

for wav_file in DATASET_PATH.rglob("*.wav"):
    parts = wav_file.stem.split("-")

    emotion_code = int(parts[2])
    actor_id = int(parts[6])

    records.append({
        "file_path": str(wav_file),
        "actor": actor_id,
        "emotion": emotion_map[emotion_code]
    })

df = pd.DataFrame(records)

print("Total audio files:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts())

df.to_csv("data/ravdess_metadata.csv", index=False)

print("\nMetadata saved successfully!")