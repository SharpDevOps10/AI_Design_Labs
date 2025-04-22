import os
import shutil
import random
import xml.etree.ElementTree as ET

# Класи та їх ID
prefix_to_class = {
    "n02099712": 0,  # labrador_retriever
    "n02106550": 1,  # rottweiler
    "n02106662": 2,  # german_shepherd
    "n02108915": 3,  # french_bulldog
    "n02109525": 4,  # saint_bernard
    "n02110185": 5,  # siberian_husky
    "n02100583": 6,  # vizsla
    "n02088364": 7,  # beagle
}

# Шляхи
src_images = "/content/drive/MyDrive/Colab Notebooks/dataset/images"
src_labels = "/content/drive/MyDrive/Colab Notebooks/dataset/labels"
dst_root = "/content/drive/MyDrive/Colab Notebooks/dataset_yolo"
train_ratio = 0.8

# Цільова структура
for split in ["train", "val"]:
    os.makedirs(f"{dst_root}/images/{split}", exist_ok=True)
    os.makedirs(f"{dst_root}/labels/{split}", exist_ok=True)

# Всі допустимі приклади (без розширення)
examples = []
for filename in os.listdir(src_labels):
    full_path = os.path.join(src_labels, filename)
    if not os.path.isfile(full_path):
        continue

    name = os.path.splitext(filename)[0]  # без .xml або інших
    prefix = name.split("_")[0]
    if prefix in prefix_to_class:
        examples.append(name)

print(f"Знайдено {len(examples)} прикладів для обробки.")

# Тренувальні/тестові розділення
random.shuffle(examples)
split_index = int(train_ratio * len(examples))
train_examples = examples[:split_index]
val_examples = examples[split_index:]


# XML → YOLO
def convert_annotation(xml_path, class_id, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines


# Обробка
for example_list, split in [(train_examples, "train"), (val_examples, "val")]:
    for name in example_list:
        prefix = name.split("_")[0]
        class_id = prefix_to_class[prefix]

        xml_path = os.path.join(src_labels, name)  # без .xml
        if not os.path.exists(xml_path):
            print(f"[!] Пропущено — не знайдено: {xml_path}")
            continue

        # Пошук зображення з розширеннями
        img_path = None
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]:
            candidate = os.path.join(src_images, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"[!] Зображення не знайдено для: {name}")
            continue

        # Отримуємо розміри
        try:
            tree = ET.parse(xml_path)
            size = tree.getroot().find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
        except Exception as e:
            print(f"⛔️ Помилка при читанні {xml_path}: {e}")
            continue

        yolo_lines = convert_annotation(xml_path, class_id, w, h)
        if not yolo_lines:
            print(f"[!] В аннотації {xml_path} немає об'єктів")
            continue

        # Запис результату
        dst_img_path = f"{dst_root}/images/{split}/{name}.jpg"
        dst_lbl_path = f"{dst_root}/labels/{split}/{name}.txt"

        shutil.copy(img_path, dst_img_path)
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("✅ Готово: зображення та анотації конвертовано.")
