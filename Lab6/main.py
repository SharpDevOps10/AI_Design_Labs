import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D, Dense
)

DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/final_data/Train/"
BATCH_SIZE = 16
IMG_SIZE = (224, 224)

train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="sparse", subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="sparse", subset="validation"
)

class_names = list(train_generator.class_indices.keys())
print("Класи:", class_names)


def create_xception_model(input_shape=(224, 224, 3), num_classes=len(class_names)):
    inputs = Input(shape=input_shape)

    # Вхідний блок (Entry Flow)
    x = Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    def xception_block(x, filters, strides=2):
        res = Conv2D(filters, (1, 1), strides=strides, padding="same")(x)

        x = SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = SeparableConv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = MaxPooling2D((3, 3), strides=strides, padding="same")(x)
        x = tf.keras.layers.add([x, res])
        return x

    # Entry Flow
    x = xception_block(x, 128)
    x = xception_block(x, 256)
    x = xception_block(x, 728)

    # Middle Flow (повторюємо 8 разів)
    for _ in range(8):
        res = x
        x = SeparableConv2D(728, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = SeparableConv2D(728, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = SeparableConv2D(728, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)

        x = tf.keras.layers.add([x, res])
        x = Activation("relu")(x)

    # Exit Flow
    x = xception_block(x, 1024)
    x = SeparableConv2D(1536, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(2048, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, x)
    return model


model = create_xception_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

EPOCHS = 30
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

model.save("logo_classifier_xception.keras")
print("Модель збережено!")

model_from_file = tf.keras.models.load_model("logo_classifier_xception.keras")
model_from_file.summary()


def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]

    plt.imshow(cv2.imread(image_path)[..., ::-1])
    plt.title(f"Передбачений клас: {class_name}")
    plt.axis("off")
    plt.show()


for _ in range(3):
    logo_path = "/content/drive/MyDrive/Colab Notebooks/final_data/Test/mercedes/"
    random_image = random.choice(os.listdir(logo_path))
    test_image = os.path.join(logo_path, random_image)

    print("Обране зображення:", test_image)
    predict_image(test_image, model_from_file)

for _ in range(3):
    logo_path = "/content/drive/MyDrive/Colab Notebooks/final_data/Test/not_mercedes/"
    random_image = random.choice(os.listdir(logo_path))
    test_image = os.path.join(logo_path, random_image)

    print("Обране зображення:", test_image)
    predict_image(test_image, model_from_file)


def predict_frame(frame, model):
    IMG_SIZE = (224, 224)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]
    confidence = prediction[class_idx]

    return class_name, confidence


def process_videos(video_paths, model):
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Файл {video_path} не знайдено!")
            continue

        video_name = os.path.basename(video_path)
        print(f"\n🎥 Аналіз відео: {video_name}\n")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        detections = []
        previous_second = -1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps
            current_minute = int(current_time // 60)
            current_second = int(current_time % 60)

            class_name, confidence = predict_frame(frame, model)

            print(f"[{current_minute} хв : {current_second} сек] -> {class_name} ({confidence:.4f})")

            if current_second != previous_second:
                detections.append((current_time, class_name, confidence))
                previous_second = current_second

        cap.release()

        if detections:
            print(f"\n✅ Появи класів у відео '{video_name}':\n")
            for time, label, conf in detections:
                print(f"{int(time // 60)} хв : {int(time % 60)} сек -> {label} ({conf:.4f})")

        print(f"\n📌 Аналіз відео '{video_name}' завершено!\n")


video_files = [
    "/content/drive/MyDrive/Colab Notebooks/mer.mp4",
    "/content/drive/MyDrive/Colab Notebooks/Volkswagen logo.mp4",
    "/content/drive/MyDrive/Colab Notebooks/h2.webm"
]

process_videos(video_files, model_from_file)
