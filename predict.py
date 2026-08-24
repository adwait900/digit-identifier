import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse

def preprocess_image(image_path):
    """Reads and preprocesses the image to match MNIST format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28 (standard for MNIST)
    img_resized = cv2.resize(img, (28, 28))
    
    # Invert colors (MNIST is white on black, user images are often black on white)
    # Check if image is mostly white (background), if so, invert.
    if np.mean(img_resized) > 127:
        img_resized = np.invert(img_resized)

    # Normalize to 0-1
    img_normalized = img_resized / 255.0
    
    # Reshape for the model (Batch Size, Height, Width)
    img_reshaped = np.reshape(img_normalized, (1, 28, 28))
    
    return img_reshaped, img_resized

def predict(image_path, model_path='models/digit_recognizer.keras'):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run src/train.py first.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        processed_img, display_img = preprocess_image(image_path)
        
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        
        print(f"-----------------------------")
        print(f"Prediction: {digit}")
        print(f"Confidence: {np.max(prediction) * 100:.2f}%")
        print(f"-----------------------------")

        plt.imshow(display_img, cmap=plt.cm.binary)
        plt.title(f"Predicted: {digit}")
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten digits.")
    parser.add_argument("image", help="Path to the image file (e.g., digits/digit1.png)")
    args = parser.parse_args()

    predict(args.image)