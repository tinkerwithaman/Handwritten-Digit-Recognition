# Project: MNIST_CNN_Digit_Recognizer
# Demonstrates a foundational deep learning model using TensorFlow/Keras.
# To run this: pip install tensorflow matplotlib numpy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

def build_and_train_cnn():
    # 1. Load and Preprocess Data (MNIST is built into Keras)
    print("Loading and preprocessing MNIST data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape data to include channel dimension (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # One-hot encode the labels (e.g., 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # 2. Build the CNN Model Architecture
    print("Building CNN model...")
    model = Sequential([
        # Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Max Pooling: reduces spatial dimensions by 2x2
        MaxPooling2D((2, 2)),
        # Flatten: converts 2D feature maps to 1D feature vector
        Flatten(),
        # Dense Hidden Layer
        Dense(128, activation='relu'),
        # Output Layer: 10 classes (digits 0-9) with Softmax for probabilities
        Dense(10, activation='softmax')
    ])

    # 3. Compile and Train the Model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model (5 epochs)...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    # 4. Evaluate the Model
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # 5. Make and Visualize a Prediction
    print("\nMaking a sample prediction...")
    sample_index = np.random.randint(0, len(x_test))
    sample_image = x_test[sample_index]
    true_label = np.argmax(y_test[sample_index])

    prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
    predicted_label = np.argmax(prediction)

    plt.imshow(sample_image.reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.show()

if __name__ == '__main__':
    build_and_train_cnn()
