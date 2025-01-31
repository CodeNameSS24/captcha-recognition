# Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Get the directory of the current Python file
current_dir = os.path.dirname(__file__)

# Construct the relative path to the dataset
data_path = os.path.join(current_dir, "..", "samples")

# Load and preprocess images
def load_images(path):
    images = []
    labels = []
    
    for filename in os.listdir(path):
        if filename.endswith(('.png', '.jpg')):
            try:
                label = filename.split('.')[0]
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (128, 64))
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_images(data_path)

# Print basic stats
print(f"Total samples: {len(X)}")
print(f"Image dimensions: {X[0].shape}")
print(f"Unique CAPTCHA patterns: {len(np.unique(y))}")

# Visualize sample images
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f'Label: {y[i]}')
    plt.axis('off')
plt.show()

# Analyze character distribution
def analyze_chars(labels):
    all_chars = ''.join(labels)
    char_dist = pd.Series(list(all_chars)).value_counts()
    
    plt.figure(figsize=(12, 4))
    char_dist.plot(kind='bar')
    plt.title('Character Distribution')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.show()
    return char_dist

char_distribution = analyze_chars(y)
print("\nCharacter Distribution:")
print(char_distribution)

# Image preprocessing improvements
def preprocess_image(img):
    # Apply thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove noise
    kernel = np.ones((2,2), np.uint8)
    denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return denoised

# Update load_images function
def load_images_enhanced(path):
    images = []
    labels = []
    
    for filename in os.listdir(path):
        if filename.endswith(('.png', '.jpg')):
            try:
                label = filename.split('.')[0]
                img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 64))
                    img = preprocess_image(img)
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return np.array(images), np.array(labels)

# Load and visualize preprocessed images
X_enhanced, y_enhanced = load_images_enhanced(data_path)

plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_enhanced[i], cmap='gray')
    plt.title(f'Label: {y_enhanced[i]}')
    plt.axis('off')
plt.show()

# Analyze character patterns
def analyze_patterns(labels):
    # Length analysis
    lengths = [len(label) for label in labels]
    
    # Character type analysis
    numeric = sum(1 for label in labels if label.isdigit())
    alpha = sum(1 for label in labels if label.isalpha())
    alphanumeric = sum(1 for label in labels if label.isalnum())
    
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Numeric CAPTCHAs: {numeric}")
    print(f"Alphabetic CAPTCHAs: {alpha}")
    print(f"Alphanumeric CAPTCHAs: {alphanumeric}")
    
    return lengths

# Save preprocessed data
lengths = analyze_patterns(y)
np.save('preprocessed_images.npy', X_enhanced)
np.save('labels.npy', y_enhanced)

# Analyze image quality metrics
def analyze_image_stats(images):
    contrast = np.std(images, axis=(1,2))
    brightness = np.mean(images, axis=(1,2))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.hist(contrast, bins=30)
    plt.title('Contrast Distribution')
    plt.subplot(1,2,2)
    plt.hist(brightness, bins=30)
    plt.title('Brightness Distribution')
    plt.show()
    
    return np.mean(contrast), np.mean(brightness)

avg_contrast, avg_brightness = analyze_image_stats(X_enhanced)
print(f"Average Contrast: {avg_contrast:.3f}")
print(f"Average Brightness: {avg_brightness:.3f}")

# Add rotation correction and skew normalization
def enhance_preprocessing(img):
    # Detect skew angle
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = 90 + angle
        
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# Apply enhanced preprocessing to dataset
X_final = np.array([enhance_preprocessing(img) for img in X_enhanced])
np.save('final_preprocessed_images.npy', X_final)

# Generate summary report with key statistics
def generate_report(X, y, X_enhanced, X_final):
    # Image quality comparison
    orig_contrast = np.mean(np.std(X, axis=(1,2)))
    enhanced_contrast = np.mean(np.std(X_enhanced, axis=(1,2)))
    final_contrast = np.mean(np.std(X_final, axis=(1,2)))
    
    print("Dataset Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Image dimensions: {X[0].shape}")
    print(f"\nPreprocessing Results:")
    print(f"Original contrast: {orig_contrast:.3f}")
    print(f"Enhanced contrast: {enhanced_contrast:.3f}")
    print(f"Final contrast: {final_contrast:.3f}")
    
    # Save comparison images
    plt.figure(figsize=(15, 3))
    for i in range(3):
        plt.subplot(1, 3, 1)
        plt.imshow(X[0], cmap='gray')
        plt.title('Original')
        plt.subplot(1, 3, 2)
        plt.imshow(X_enhanced[0], cmap='gray')
        plt.title('Enhanced')
        plt.subplot(1, 3, 3)
        plt.imshow(X_final[0], cmap='gray')
        plt.title('Final')
    plt.show()

generate_report(X, y, X_enhanced, X_final)