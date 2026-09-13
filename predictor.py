import numpy as np
import librosa
import tensorflow as tf


# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "models/best_lstm.keras"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LENGTH = 228

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)

print("LSTM model loaded successfully!")


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    # Extract 40 MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    # Convert:
    # (40, time) → (time, 40)
    mfcc = mfcc.T

    # Create fixed-size sequence
    X = np.zeros(
        (MAX_LENGTH, N_MFCC),
        dtype=np.float32
    )

    length = min(len(mfcc), MAX_LENGTH)

    X[:length] = mfcc[:length]

    # Add batch dimension
    X = np.expand_dims(X, axis=0)

    return X

# ==============================
# PREDICT EMOTION
# ==============================

def predict_emotion(file_path):

    features = extract_features(file_path)

    predictions = model.predict(
        features,
        verbose=0
    )

    scores = predictions[0]

    predicted_index = np.argmax(scores)

    emotion = EMOTIONS[predicted_index]

    confidence = scores[predicted_index] * 100

    # Create emotion-score dictionary
    emotion_scores = {
        EMOTIONS[i]: float(scores[i] * 100)
        for i in range(len(EMOTIONS))
    }

    return emotion, confidence, emotion_scores


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    audio_file = input(
        "Enter path to audio file: "
    ).strip('"')

    emotion, confidence, emotion_scores = predict_emotion(
        audio_file
    )

    print("\n==============================")
    print("SPEECH EMOTION PREDICTION")
    print("==============================")

    print(f"Predicted Emotion : {emotion}")
    print(f"Model Score       : {confidence:.2f}%")

    print("\nEmotion Scores")
    print("------------------------------")

    # Sort emotions from highest to lowest score
    sorted_scores = sorted(
        emotion_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for emotion_name, score in sorted_scores:
        print(f"{emotion_name:<12} : {score:.2f}%")

    print("==============================")