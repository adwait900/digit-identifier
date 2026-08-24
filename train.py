import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import os
import numpy as np

def train_model():
    print("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 1. Reshape for CNN
    # CNNs expect (Batch, Height, Width, Color_Channels) -> (60000, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # 2. Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 3. Build a Professional CNN Model
    print("Building CNN model...")
    model = tf.keras.models.Sequential([
        # First Convolution Layer (Detects edges and curves)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second Convolution Layer (Detects complex shapes)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Dropout helps prevent overfitting (memorizing the training data)
        tf.keras.layers.Dropout(0.2),
        
        # Output Layer
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 4. Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 5. Train (Increased epochs to 5 for better learning)
    print("Starting training...")
    model.fit(x_train, y_train, epochs=5)

    # 6. Evaluate
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # 7. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save('models/digit_recognizer.keras')
    print("Model saved to models/digit_recognizer.keras")

if __name__ == "__main__":
    train_model()