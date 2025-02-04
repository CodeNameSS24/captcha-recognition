# test_model.py

# Required libraries
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
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

# Load the trained model
model_path = os.path.join(current_dir, '../trained_model.h5')
model = load_model(model_path)
print("Model loaded successfully!")

# Evaluate the model on the entire dataset or a new dataset
loss, accuracy = model.evaluate(X, y_categorical)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Example of predicting the label for a sample image
sample_image = X[0].reshape(1, 64, 128, 1)  # Reshape to (1, height, width, 1)
predicted_label = model.predict(sample_image)
predicted_label = unique_labels[np.argmax(predicted_label)]  # Convert to label

print(f"Predicted label for the sample image: {predicted_label}")
