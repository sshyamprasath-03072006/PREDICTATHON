import numpy as np
import os
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set dataset paths
DATASET_DIR = "./DATASET"
REAL_DIR = os.path.join(DATASET_DIR, "real_cifake_images")
FAKE_DIR = os.path.join(DATASET_DIR, "fake_cifake_images")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image settings
IMG_SIZE = (128, 128)  
X, y = [], []

# Function to load images
def load_images_from_folder(folder, label):
    for img_path in Path(folder).glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0  # Normalize
            X.append(img)
            y.append(label)

# Load images
load_images_from_folder(REAL_DIR, 0)  # 0 = real
load_images_from_folder(FAKE_DIR, 1)  # 1 = fake

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🏗 Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save trained model
model.save("deepfake_detector.h5")
print("✅ Model trained and saved as deepfake_detector.h5")

# 🧐 Predicting on test images
def predict_on_test_images():
    test_images = []
    image_paths = []
    
    for img_path in sorted(Path(TEST_DIR).glob("*.*")):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, IMG_SIZE) / 255.0
            test_images.append(img)
            image_paths.append(img_path)
    
    test_images = np.array(test_images)
    predictions = (model.predict(test_images) > 0.5).astype("int32")  # Convert to 0 or 1
    
    results = []
    for index, pred in enumerate(predictions):
        results.append({"index": index + 1, "prediction": "fake" if pred[0] == 1 else "real"})
    
    # Save to JSON
    output_file = "HACKONAUTZ_prediction.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Predictions saved in {output_file}")

# Run predictions
predict_on_test_images()
