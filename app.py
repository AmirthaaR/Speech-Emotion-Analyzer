import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tempfile
import os


# ==============================
# CONFIGURATION
# ==============================

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LENGTH = 228

MODEL_PATH = "models/best_lstm.keras"

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
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Speech Emotion Analyzer",
    page_icon="🎙️",
    layout="wide"
)

st.title("Speech Emotion Analyzer")
st.write(
    "Upload a speech recording to analyze its emotional characteristics "
    "using a deep learning LSTM model."
)


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T

    X = np.zeros(
        (MAX_LENGTH, N_MFCC),
        dtype=np.float32
    )

    length = min(len(mfcc), MAX_LENGTH)

    X[:length] = mfcc[:length]

    X = np.expand_dims(X, axis=0)

    return audio, sr, mfcc, X


# ==============================
# PREDICTION
# ==============================

def predict_emotion(features):

    predictions = model.predict(
        features,
        verbose=0
    )

    scores = predictions[0]

    predicted_index = np.argmax(scores)

    emotion = EMOTIONS[predicted_index]

    return emotion, scores


# ==============================
# AUDIO UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "mpeg", "ogg", "flac"]
)


if uploaded_file is not None:

    st.audio(uploaded_file)

    # Save uploaded file temporarily
    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())

        temp_path = temp_file.name

    try:

        audio, sr, mfcc, features = extract_features(temp_path)

        duration = len(audio) / sr

        st.success("Audio loaded successfully!")

        st.write(f"Duration: {duration:.2f} seconds")
        st.write(f"Sample Rate: {sr} Hz")

        st.divider()

        # ==============================
        # VISUALIZATION
        # ==============================

        st.subheader("Audio Visualization")

        col1, col2 = st.columns(2)

        with col1:

            fig_wave, ax_wave = plt.subplots(figsize=(8, 3))

            librosa.display.waveshow(
                audio,
                sr=sr,
                ax=ax_wave
            )

            ax_wave.set_title("Speech Waveform")
            ax_wave.set_xlabel("Time (seconds)")
            ax_wave.set_ylabel("Amplitude")

            st.pyplot(fig_wave)

            plt.close(fig_wave)

        with col2:

            D = librosa.stft(audio)

            spectrogram = librosa.amplitude_to_db(
                np.abs(D),
                ref=np.max
            )

            fig_spec, ax_spec = plt.subplots(figsize=(8, 3))

            img_spec = librosa.display.specshow(
                spectrogram,
                sr=sr,
                x_axis="time",
                y_axis="hz",
                ax=ax_spec
            )

            ax_spec.set_title("Speech Spectrogram")

            fig_spec.colorbar(
                img_spec,
                ax=ax_spec
            )
            st.pyplot(fig_spec)

            plt.close(fig_spec)

        fig_mfcc, ax_mfcc = plt.subplots(figsize=(12, 4))

        img_mfcc = librosa.display.specshow(
            mfcc.T,
            sr=sr,
            x_axis="time",
            ax=ax_mfcc
        )

        ax_mfcc.set_title("MFCC Features")
        ax_mfcc.set_xlabel("Time (seconds)")
        ax_mfcc.set_ylabel("MFCC Coefficient")

        fig_mfcc.colorbar(
            img_mfcc,
            ax=ax_mfcc
        )

        st.pyplot(fig_mfcc)

        plt.close(fig_mfcc)

        st.divider()

        # ==============================
        # EMOTION PREDICTION
        # ==============================

        st.subheader("Emotion Prediction")

        emotion, scores = predict_emotion(features)

        predicted_index = np.argmax(scores)

        confidence = scores[predicted_index] * 100

        st.metric(
            label="Dominant Emotion",
            value=emotion.upper()
        )

        st.write(
            f"Model Score: {confidence:.2f}%"
        )

        st.divider()

        # ==============================
        # EMOTION SCORE DISTRIBUTION
        # ==============================

        st.subheader("Emotion Score Distribution")

        emotion_scores = {
            EMOTIONS[i]: float(scores[i] * 100)
            for i in range(len(EMOTIONS))
        }

        sorted_scores = dict(
            sorted(
                emotion_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        st.bar_chart(sorted_scores)

        st.table(
            {
                "Emotion": list(sorted_scores.keys()),
                "Score (%)": [
                    round(score, 2)
                    for score in sorted_scores.values()
                ]
            }
        )

        # ==============================
        # INTERPRETATION
        # ==============================

        second_highest = sorted_scores[
            list(sorted_scores.keys())[1]
        ]

        second_score = sorted_scores[
            list(sorted_scores.keys())[1]
        ]

        st.info(
            f"The dominant predicted emotion is {emotion.upper()}. "
            f"The model's highest score is {confidence:.2f}%. "
            f"Because the prediction score is not very high, the result "
            f"should be interpreted as a model estimate rather than certainty."
        )

    except Exception as e:

        st.error(f"Error while processing audio: {e}")

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)