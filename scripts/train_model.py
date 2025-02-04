# train_model.py

# Required libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Load preprocessed data
current_dir = os.path.dirname(__file__)
X = np.load(os.path.join(current_dir, '../final_preprocessed_images.npy'))
y = np.load(os.path.join(current_dir, '../labels.npy'))

# Convert labels to categorical (one-hot encoding)
unique_labels = sorted(set(y))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_numeric = np.array([label_to_index[label] for label in y])
y_categorical = to_categorical(y_numeric, num_classes=len(unique_labels))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Expand dimensions for compatibility with CNN input
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (samples, height, width, 1)
X_val = np.expand_dims(X_val, axis=-1)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(unique_labels), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
model.save(os.path.join(current_dir, '../trained_model.h5'))
print("Model saved as trained_model.h5")

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
