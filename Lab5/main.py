import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import concatenate

dataset_path = "/content/dogs/dog-breeds"

batch_size = 32
img_size = (299, 299)

train_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Класи: {class_names}")


def preprocess(image, label):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label


train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)


def InceptionBlock(x, filters):
    f1, f3r, f3, f5r, f5, proj = filters

    conv1x1_1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    conv3x3 = Conv2D(f3r, (1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(f3, (3, 3), padding='same', activation='relu')(conv3x3)

    conv5x5 = Conv2D(f5r, (1, 1), padding='same', activation='relu')(x)
    conv5x5 = Conv2D(f5, (5, 5), padding='same', activation='relu')(conv5x5)
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    maxpool = Conv2D(proj, (1, 1), padding='same', activation='relu')(maxpool)

    output = concatenate([conv1x1_1, conv3x3, conv5x5, maxpool], axis=-1)
    return output


def InceptionV3Custom(num_classes):
    input_layer = Input(shape=(299, 299, 3))

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(80, (1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = InceptionBlock(x, (64, 48, 64, 64, 96, 32))
    x = InceptionBlock(x, (64, 48, 64, 64, 96, 64))
    x = InceptionBlock(x, (64, 48, 64, 64, 96, 64))

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=predictions)
    return model


model = InceptionV3Custom(num_classes)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

epochs = 50
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


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

model.save("inception_v3_custom.h5")


def predict_image(image_path):
    model = tf.keras.models.load_model("inception_v3_custom.h5")

    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Клас: {predicted_label}")
    plt.show()

    print(f"Передбачена порода: {predicted_label}")


predict_image("/content/drive/MyDrive/Colab Notebooks/dogs/PXL_20250128_174735529.jpg")
predict_image("/content/drive/MyDrive/Colab Notebooks/dogs/working-german-shepherds-as-pets-and-companions.jpg")
predict_image("/content/drive/MyDrive/Colab Notebooks/dogs/n02106550_107.jpg")
