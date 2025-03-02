import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def preprocess(image, label):
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label


dataset, info = tfds.load("stanford_dogs", split=["train", "test"], as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset

train_dataset = train_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(info.features["label"].num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

epochs = 5
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Графік втрат')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Графік точності')

    plt.show()


plot_training_history(history)

model.save("dog_classifier.h5")


def predict_and_display_images(directory):
    model = tf.keras.models.load_model("dog_classifier.h5")
    class_names = info.features["label"].names

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))][:9]

    for i, img_path in enumerate(image_files):
        img = load_img(img_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Клас: {predicted_label}")

    plt.show()


img_dir = "/content/drive/MyDrive/Colab Notebooks/dogs"
predict_and_display_images(img_dir)
