import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ==============================
# CONFIGURATION
# ==============================

SAMPLE_RATE = 22050
N_MFCC = 40


# ==============================
# LOAD AUDIO
# ==============================

audio_file = input(
    "Enter path to audio file: "
).strip('"')

audio, sr = librosa.load(
    audio_file,
    sr=SAMPLE_RATE
)

print("\nAudio loaded successfully!")
print(f"Duration: {len(audio) / sr:.2f} seconds")
print(f"Sample Rate: {sr} Hz")


# ==============================
# 1. WAVEFORM
# ==============================

plt.figure(figsize=(12, 4))

librosa.display.waveshow(
    audio,
    sr=sr
)

plt.title("Speech Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()


# ==============================
# 2. SPECTROGRAM
# ==============================

D = librosa.stft(audio)

spectrogram = librosa.amplitude_to_db(
    np.abs(D),
    ref=np.max
)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    spectrogram,
    sr=sr,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Speech Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()
plt.show()


# ==============================
# 3. MFCC
# ==============================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=N_MFCC
)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mfcc,
    sr=sr,
    x_axis="time"
)

plt.colorbar()
plt.title("MFCC Features")
plt.xlabel("Time (seconds)")
plt.ylabel("MFCC Coefficient")

plt.tight_layout()
plt.show()


print("\nAudio visualization completed!")