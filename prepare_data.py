import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Load extracted features
df = pd.read_csv("data/features.csv")

print("Original dataset shape:", df.shape)


# Separate features and labels
feature_columns = [f"feature_{i}" for i in range(1, 81)]

X = df[feature_columns].values
y = df["emotion"].values
actors = df["actor"].values


# Encode emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEmotion classes:")
for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i}: {emotion}")


# --------------------------------------------------
# Actor-independent split
# --------------------------------------------------

# Actors 01-20 -> training/validation
# Actors 21-24 -> testing

train_val_mask = actors <= 20
test_mask = actors > 20

X_train_val = X[train_val_mask]
y_train_val = y_encoded[train_val_mask]

X_test = X[test_mask]
y_test = y_encoded[test_mask]


# Split training/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val
)


# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


print("\nData split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nPreparation completed successfully!")

# Save prepared datasets
np.save("data/X_train.npy", X_train)
np.save("data/X_val.npy", X_val)
np.save("data/X_test.npy", X_test)

np.save("data/y_train.npy", y_train)
np.save("data/y_val.npy", y_val)
np.save("data/y_test.npy", y_test)

print("Prepared datasets saved successfully!")
