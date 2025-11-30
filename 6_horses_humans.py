import tensorflow as tf
import tensorflow_datasets as tfds # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# a️⃣ Load the dataset using TFDS (TADS)
# ---------------------------------------------------------------------
# The 'horses_or_humans' dataset is built into TensorFlow Datasets
dataset_name = 'horses_or_humans'

# Load with splits
(train_ds, test_ds), ds_info = tfds.load(
    dataset_name,
    split=['train[:80%]', 'train[80%:]'],  # 80/20 split
    as_supervised=True,  # (image, label)
    with_info=True
)

print("✅ Dataset loaded successfully via TensorFlow Datasets!")

# ---------------------------------------------------------------------
# b️⃣ View number of training and testing images
# ---------------------------------------------------------------------
train_count = ds_info.splits['train'].num_examples * 0.8
test_count = ds_info.splits['train'].num_examples * 0.2
print(f"Training images: {int(train_count)}")
print(f"Testing images: {int(test_count)}")

# ---------------------------------------------------------------------
# c️⃣ Plot some images
# ---------------------------------------------------------------------
plt.figure(figsize=(10,5))
for i, (image, label) in enumerate(train_ds.take(6)):
    plt.subplot(2, 3, i+1)
    plt.imshow(image)
    plt.title("Horse" if label == 1 else "Human")
    plt.axis('off')
plt.show()

# ---------------------------------------------------------------------
# d️⃣ Normalize the training data (rescale 0–255 → 0–1)
# ---------------------------------------------------------------------
def normalize_img(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize for ResNet input
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize_img).batch(32).shuffle(1000)
test_ds = test_ds.map(normalize_img).batch(32)

# ---------------------------------------------------------------------
# e️⃣ Build a ResNet model (Transfer Learning)
# ---------------------------------------------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze pre-trained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ---------------------------------------------------------------------
# f️⃣ Train the model
# ---------------------------------------------------------------------
history = model.fit(train_ds, validation_data=test_ds, epochs=5)

# ---------------------------------------------------------------------
# g️⃣ Show the training and testing accuracy
# ---------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.show()

print(f"✅ Final Training Accuracy: {history.history['accuracy'][-1]:.2f}")
print(f"✅ Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}")