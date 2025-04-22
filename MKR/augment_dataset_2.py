# pip install albumentations opencv-python
import os
import cv2
import albumentations as A

# Параметри
input_dir = "/content/drive/MyDrive/Colab Notebooks/dataset_yolo"
output_dir = "/content/drive/MyDrive/Colab Notebooks/dataset_yolo_aug"
num_aug_per_image = 3


# YOLO -> Pascal VOC
def yolo_to_voc(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = x * img_w
    y_center = y * img_h
    w *= img_w
    h *= img_h
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    return [x_min, y_min, x_max, y_max]


# Pascal VOC -> YOLO + кліп до [0.0, 1.0]
def voc_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox

    x_min = max(0.0, min(x_min, img_w - 1))
    y_min = max(0.0, min(y_min, img_h - 1))
    x_max = max(0.0, min(x_max, img_w - 1))
    y_max = max(0.0, min(y_max, img_h - 1))

    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return [
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    ]


# Кліп bounding box у VOC-форматі
def clip_voc_bbox(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    return [
        max(0.0, min(x_min, img_w - 1)),
        max(0.0, min(y_min, img_h - 1)),
        max(0.0, min(x_max, img_w - 1)),
        max(0.0, min(y_max, img_h - 1)),
    ]


# Створення директорій
for split in ["train", "val"]:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

# Pipline аугментацій
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),  # важливо
    A.Blur(blur_limit=3, p=0.1),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.2)
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels'],
    min_visibility=0.01,
    check_each_transform=True
))

bad_boxes_total = 0
skipped_augs = 0

for split in ["train", "val"]:
    img_dir = os.path.join(input_dir, "images", split)
    lbl_dir = os.path.join(input_dir, "labels", split)

    for fname in os.listdir(img_dir):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        if image is None or not os.path.exists(lbl_path):
            continue

        h, w = image.shape[:2]

        with open(lbl_path, "r") as f:
            lines = f.read().strip().split("\n")

        boxes_yolo = []
        class_labels = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = int(parts[0])
            bbox = list(map(float, parts[1:]))
            if all(0.0 <= b <= 1.0 for b in bbox):
                boxes_yolo.append(bbox)
                class_labels.append(cls)
            else:
                bad_boxes_total += 1

        if not boxes_yolo:
            continue

        # YOLO -> VOC + clipping
        boxes_voc = [clip_voc_bbox(yolo_to_voc(b, w, h), w, h) for b in boxes_yolo]

        # Зберігаємо оригінал
        cv2.imwrite(os.path.join(output_dir, "images", split, fname), image)
        with open(os.path.join(output_dir, "labels", split, fname.replace(".jpg", ".txt")), "w") as f:
            for cls, b in zip(class_labels, boxes_yolo):
                f.write(f"{int(cls)} {' '.join(f'{v:.6f}' for v in b)}\n")

        # Аугментація
        for i in range(num_aug_per_image):
            try:
                augmented = transform(image=image, bboxes=boxes_voc, class_labels=class_labels)
                aug_img = augmented["image"]
                aug_boxes_voc = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                if not aug_boxes_voc:
                    skipped_augs += 1
                    continue

                # VOC -> YOLO + clipping
                aug_boxes_yolo = [voc_to_yolo(b, aug_img.shape[1], aug_img.shape[0]) for b in aug_boxes_voc]

                aug_name = fname.replace(".jpg", f"_aug{i}.jpg")
                cv2.imwrite(os.path.join(output_dir, "images", split, aug_name), aug_img)
                with open(os.path.join(output_dir, "labels", split, aug_name.replace(".jpg", ".txt")), "w") as f:
                    for cls, b in zip(aug_labels, aug_boxes_yolo):
                        f.write(f"{int(cls)} {' '.join(f'{v:.6f}' for v in b)}\n")

            except Exception as e:
                print(f"[!] Помилка при аугментації {fname} (i={i}): {e}")
                skipped_augs += 1

print("✅ Аугментація завершена.")
print(f"❗️ Пропущено невалідних боксів: {bad_boxes_total}")
print(f"⚠️ Пропущено аугментацій через помилки: {skipped_augs}")
print(f"📁 Результат збережено у {output_dir}/")
