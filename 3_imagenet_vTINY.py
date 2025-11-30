# Dataset: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
# CNN Accuracy: 54.60%



import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils
from sklearn.utils import shuffle

# Configuration
DATA_DIR = '/home/narayan/Documents/code/image-processing/archive/tiny-imagenet-200'
WNIDS_PATH = os.path.join(DATA_DIR, 'wnids.txt')
WORDS_PATH = os.path.join(DATA_DIR, 'words.txt')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
NUM_CLASSES_TO_USE = 20  # Use a subset of classes to prevent OOM
IMG_SIZE = 64

def load_data(num_classes=20):
    print(f"Loading data for first {num_classes} classes...")
    
    # Read wnids
    with open(WNIDS_PATH, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    
    selected_wnids = wnids[:num_classes]
    wnid_to_label = {wnid: i for i, wnid in enumerate(selected_wnids)}
    
    # Load Train Data
    X_train = []
    y_train = []
    
    print("Loading training images...")
    for wnid in selected_wnids:
        class_dir = os.path.join(TRAIN_DIR, wnid, 'images')
        if not os.path.exists(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_train.append(img)
                y_train.append(wnid_to_label[wnid])
                
    # Load Val Data (as Test)
    X_test = []
    y_test = []
    
    print("Loading validation (test) images...")
    val_img_dir = os.path.join(VAL_DIR, 'images')
    val_annot_path = os.path.join(VAL_DIR, 'val_annotations.txt')
    
    val_img_to_wnid = {}
    with open(val_annot_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            val_img_to_wnid[parts[0]] = parts[1]
            
    for img_name in os.listdir(val_img_dir):
        wnid = val_img_to_wnid.get(img_name)
        if wnid in selected_wnids:
            img_path = os.path.join(val_img_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X_test.append(img)
                y_test.append(wnid_to_label[wnid])
                
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), selected_wnids

def augment_image(image):
    """Apply contrast, flipping, and rotation."""
    augmented_images = []
    
    # Original
    # augmented_images.append(image) # We keep original in the main list
    
    # Flip
    flip = cv2.flip(image, 1) # Horizontal flip
    augmented_images.append(flip)
    
    # Rotation (e.g., 15 degrees)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rot = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(rot)
    
    # Contrast
    # Convert to PIL or use cv2
    # alpha = 1.5 (contrast), beta = 0 (brightness)
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
    X_train, y_train, X_test, y_test, classes = load_data(NUM_CLASSES_TO_USE)
    
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
        plt.title(f"Class: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved to 'sample_images.png'")
    
    # d. Do the image augmentation
    print("\nStep d: Performing image augmentation (Contrast, Flip, Rotate)...")
    X_train_aug = []
    y_train_aug = []
    
    # We will augment a subset if it's too large, but for 20 classes it should be fine.
    # Original data is kept, plus augmented versions.
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
    cnn_model = build_cnn((IMG_SIZE, IMG_SIZE, 3), len(classes))
    cnn_model.summary()
    
    print("Training CNN...")
    history_cnn = cnn_model.fit(X_train_aug, y_train_aug, epochs=5, batch_size=64, validation_data=(X_test, y_test))
    
    # h. Show the training and testing accuracy
    print("\nStep h: CNN Results")
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Testing Accuracy: {cnn_acc*100:.2f}%")
    print(f"CNN Training Accuracy (Final Epoch): {history_cnn.history['accuracy'][-1]*100:.2f}%")
    

    


if __name__ == "__main__":
    main()
