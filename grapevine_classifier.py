import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import os

tf.random.set_seed(42)
# a. Load the dataset
dataset_dir = pathlib.Path("Grapevine_Leaves_Image_Dataset")
image_count = len(list(dataset_dir.glob('*/*.png')))
print(f"Total images found: {image_count}")

batch_size = 32
img_height = 180
img_width = 180

print("\n--- Loading Datasets ---")
# b. Show the no. of testing and training images
train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# c. Plot some images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.savefig("original_samples.png")
print("original_samples.png")



# d. Do the image augmentation – contrast, flipping and rotation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.5),
  tf.keras.layers.RandomContrast(0.5),
])

# Visualize augmentation
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.savefig("augmented_samples.png")
print("augmented_samples.png")

# e. After augmentation, show the no. of testing and training images
print("\n--- Data Augmentation Info ---")
print("Note: Keras preprocessing layers apply augmentation dynamically during training.")
print("The physical number of images remains the same, but the model sees slightly different versions each epoch.")
count = 0
for images, labels in train_ds:
    count += images.shape[0]

print("Total images in train_ds:", count)

count = 0
for images, labels in val_ds:
    count += images.shape[0]

print("Total images in val_ds:", count)

# f. Normalizing the training data
normalization_layer = tf.keras.layers.Rescaling(1./255)

# g. Build a convolutional neural network to train images (Model 1: No Augmentation)
print("\n--- Training Model 1 (No Augmentation) ---")
num_classes = len(class_names)

model_no_aug = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model_no_aug.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
history_no_aug = model_no_aug.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  verbose=1
)

# h. Show the training and testing accuracy
acc = history_no_aug.history['accuracy']
val_acc = history_no_aug.history['val_accuracy']
print(f"Model 1 Final Training Accuracy: {acc[-1]:.4f}")
print(f"Model 1 Final Validation Accuracy: {val_acc[-1]:.4f}")

# i. Build a convolutional neural network to train images (Model 2: With Augmentation)
print("\n--- Training Model 2 (With Augmentation) ---")
model_aug = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model_aug.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_aug = model_aug.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  verbose=1
)

# j. Show the training and testing accuracy
acc_aug = history_aug.history['accuracy']
val_acc_aug = history_aug.history['val_accuracy']
print(f"Model 2 Final Training Accuracy: {acc_aug[-1]:.4f}")
print(f"Model 2 Final Validation Accuracy: {val_acc_aug[-1]:.4f}")

# k. Compare the training and testing accuracy before and after augmentation
print("\n--- Comparison Results ---")
print(f"{'Metric':<25} | {'No Augmentation':<15} | {'With Augmentation':<15}")
print("-" * 60)
print(f"{'Training Accuracy':<25} | {acc[-1]:.4f}          | {acc_aug[-1]:.4f}")
print(f"{'Validation Accuracy':<25} | {val_acc[-1]:.4f}          | {val_acc_aug[-1]:.4f}")


