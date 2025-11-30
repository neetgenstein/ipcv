# CNN Training Acc ~19%
# CNN Testing Acc ~0.55%


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32

def load_data():
    print("Loading imagenet_v2 dataset...")
    # imagenet_v2 only has a 'test' split. We will use it and split it manually.
    ds, info = tfds.load("imagenet_v2", split="test", as_supervised=True, with_info=True)
    
    # Convert dataset to numpy arrays for easier manipulation with the existing pipeline structure
    # Note: imagenet_v2 is relatively small (10k images), so this fits in memory.
    images = []
    labels = []
    
    print("Converting dataset to numpy arrays...")
    for img, label in tfds.as_numpy(ds):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)
        
    X = np.array(images)
    y = np.array(labels)
    
    # Split into train and test (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test, info

def augment_image(image):
    """Apply contrast, flipping, and rotation."""
    augmented_images = []
    
    # Flip
    flip = cv2.flip(image, 1) # Horizontal flip
    augmented_images.append(flip)
    
    # Rotation (e.g., 15 degrees)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rot = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(rot)
    
    # Contrast
    contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    augmented_images.append(contrast)
    
    return augmented_images

def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def main():
    # a. Load the dataset
    X_train, y_train, X_test, y_test, info = load_data()
    num_classes = info.features['label'].num_classes
    
    # b. Show the no. of testing and training images
    print(f"\nStep b: Dataset Statistics (Before Augmentation)")
    print(f"Training images: {len(X_train)}")
    print(f"Testing images: {len(X_test)}")
    
    # c. Plot some images
    print("\nStep c: Plotting sample images...")
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_train[i])
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images_v2.png')
    print("Sample images saved to 'sample_images_v2.png'")
    
    # d. Do the image augmentation
    print("\nStep d: Performing image augmentation (Contrast, Flip, Rotate)...")
    X_train_aug = []
    y_train_aug = []
    
    # Augmenting a subset to keep processing time reasonable if needed, 
    # but for 8k images, we can augment all.
    for i in range(len(X_train)):
        img = X_train[i]
        label = y_train[i]
        
        # Add original
        X_train_aug.append(img)
        y_train_aug.append(label)
        
        # Add augmented
        aug_imgs = augment_image(img)
        for aug_img in aug_imgs:
            X_train_aug.append(aug_img)
            y_train_aug.append(label)
            
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    # e. After augmentation, show the no. of testing and training images
    print(f"\nStep e: Dataset Statistics (After Augmentation)")
    print(f"Training images: {len(X_train_aug)}")
    print(f"Testing images: {len(X_test)}")
    
    # f. Normalizing the training data
    print("\nStep f: Normalizing data...")
    X_train_aug = X_train_aug.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # g. Build a convolutional neural network
    print("\nStep g: Building CNN...")
    cnn_model = build_cnn((IMG_SIZE, IMG_SIZE, 3), num_classes)
    cnn_model.summary()
    
    print("Training CNN...")
    # Reduced epochs for demonstration speed
    history_cnn = cnn_model.fit(X_train_aug, y_train_aug, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    # h. Show the training and testing accuracy
    print("\nStep h: CNN Results")
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Testing Accuracy: {cnn_acc*100:.2f}%")
    print(f"CNN Training Accuracy (Final Epoch): {history_cnn.history['accuracy'][-1]*100:.2f}%")
    

    


if __name__ == "__main__":
    main()
