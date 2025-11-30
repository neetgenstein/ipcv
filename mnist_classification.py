import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print("--- MNIST Classification ---")

# a. Load the dataset
print("\n[Step a] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# b. View the no. of testing and training images
print("\n[Step b] Dataset statistics:")
print(f"Training images: {X_train.shape[0]}")
print(f"Testing images: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]}")

# c. Plot some images
print("\n[Step c] Plotting sample images...")
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.savefig("images.png")
print("Images plotted.")

# d. Normalizing the training data
print("\n[Step d] Normalizing data...")
X_train = X_train / 255.0
X_test = X_test / 255.0

# e. Build simple artificial neural network (ANN)
print("\n[Step e] Building and training ANN...")
ann = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

print("Training ANN...")
ann_history = ann.fit(X_train, y_train, epochs=3)

# f. Build a convolutional neural network (CNN)
print("\n[Step f] Building and training CNN...")
# Reshape for CNN (batch, height, width, channels)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

cnn = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

print("Training CNN...")
cnn_history = cnn.fit(X_train_cnn, y_train, epochs=3)

# g. Show the training and testing accuracy
print("\n[Step g] Evaluation Results:")

print("\nANN Evaluation:")
ann_loss, ann_acc = ann.evaluate(X_test, y_test, verbose=0)
print(f"ANN Train Accuracy: {ann_history.history['accuracy'][-1]:.4f}")
print(f"ANN Test Accuracy: {ann_acc:.4f}")

print("\nCNN Evaluation:")
cnn_loss, cnn_acc = cnn.evaluate(X_test_cnn, y_test, verbose=0)
print(f"CNN Train Accuracy: {cnn_history.history['accuracy'][-1]:.4f}")
print(f"CNN Test Accuracy: {cnn_acc:.4f}")


