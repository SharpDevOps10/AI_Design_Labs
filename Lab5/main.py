import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# --- 1. Перейменування директорій (італійська -> англійська) ---
translate_dict = {
    "cane": "dog", "gatto": "cat", "cavallo": "horse", "ragno": "spider",
    "farfalla": "butterfly", "gallina": "chicken", "pecora": "sheep",
    "mucca": "cow", "scoiattolo": "squirrel", "elefante": "elephant"
}

dataset_path = "/content/animals10/raw-img"

for italian_name, english_name in translate_dict.items():
    italian_path = os.path.join(dataset_path, italian_name)
    english_path = os.path.join(dataset_path, english_name)
    if os.path.exists(italian_path):
        os.rename(italian_path, english_path)
        print(f"Renamed {italian_name} -> {english_name}")

# --- 2. Розділення на TRAIN / VAL директорії ---
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")

if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in translate_dict.values():
        src_path = os.path.join(dataset_path, category)
        train_dst = os.path.join(train_dir, category)
        val_dst = os.path.join(val_dir, category)

        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(val_dst, exist_ok=True)

        files = os.listdir(src_path)
        np.random.shuffle(files)
        split_idx = int(0.8 * len(files))  # 80% тренування, 20% валідація

        # 80% -> TRAIN
        for file in files[:split_idx]:
            shutil.move(os.path.join(src_path, file), os.path.join(train_dst, file))

        # 20% -> VAL
        for file in files[split_idx:]:
            shutil.move(os.path.join(src_path, file), os.path.join(val_dst, file))

    print("✅ Розділення на train та val завершено!")
else:
    print("✅ Train/Val директорії вже існують.")

# --- 3. Завантаження датасетів ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=(227, 227), batch_size=128, label_mode="categorical"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=(227, 227), batch_size=128, label_mode="categorical"
)

class_names = train_ds.class_names  # Зберігаємо назви класів

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)


# --- 4. Модель AlexNet ---
def alexnet_model(input_shape=(227, 227, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),

        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),

        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# --- 5. Модель EfficientNetB0 ---
def efficientnet_model(input_shape=(227, 227, 3), num_classes=10):
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Заморожуємо базову модель
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


# --- 6. Навчання AlexNet ---
model_alexnet = alexnet_model()
model_alexnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_alexnet = model_alexnet.fit(
    train_dataset, epochs=30, validation_data=val_dataset
)

# --- 7. Навчання EfficientNetB0 ---
model_efficientnet = efficientnet_model()
model_efficientnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_efficientnet = model_efficientnet.fit(
    train_dataset, epochs=30, validation_data=val_dataset
)

# --- 8. Оцінка моделей ---
test_loss_alexnet, test_acc_alexnet = model_alexnet.evaluate(val_dataset)
print(f'✅ AlexNet Test accuracy: {test_acc_alexnet}')

test_loss_efficientnet, test_acc_efficientnet = model_efficientnet.evaluate(val_dataset)
print(f'✅ EfficientNet Test accuracy: {test_acc_efficientnet}')


# --- 9. Ансамбль передбачень ---
def ensemble_predict(image):
    pred_alexnet = model_alexnet.predict(image)
    pred_efficientnet = model_efficientnet.predict(image)
    final_pred = (pred_alexnet + pred_efficientnet) / 2  # Усереднення
    return np.argmax(final_pred, axis=1)


# --- 10. Візуалізація точності та втрат ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_alexnet.history['accuracy'], label='AlexNet Train Accuracy')
plt.plot(history_alexnet.history['val_accuracy'], label='AlexNet Val Accuracy')
plt.plot(history_efficientnet.history['accuracy'], label='EfficientNet Train Accuracy')
plt.plot(history_efficientnet.history['val_accuracy'], label='EfficientNet Val Accuracy')
plt.legend()
plt.title("Accuracy Comparison")

plt.subplot(1, 2, 2)
plt.plot(history_alexnet.history['loss'], label='AlexNet Train Loss')
plt.plot(history_alexnet.history['val_loss'], label='AlexNet Val Loss')
plt.plot(history_efficientnet.history['loss'], label='EfficientNet Train Loss')
plt.plot(history_efficientnet.history['val_loss'], label='EfficientNet Val Loss')
plt.legend()
plt.title("Loss Comparison")

plt.show()

# --- 11. Демонстрація ансамбль передбачень на тестових зображеннях ---
test_images, test_labels = next(iter(val_dataset))

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].numpy().astype("uint8"))

    predicted_label = ensemble_predict(np.expand_dims(test_images[i], axis=0))[0]
    true_label = np.argmax(test_labels[i])

    true_class = class_names[true_label]
    predicted_class = class_names[predicted_label]

    plt.title(f"True: {true_class}\nPred: {predicted_class}")
    plt.axis('off')

plt.show()

# --- 12. Збереження моделей ---
model_alexnet.save('alexnet_animals10.h5')
model_efficientnet.save('efficientnet_animals10.h5')

print("✅ Моделі збережено!")
