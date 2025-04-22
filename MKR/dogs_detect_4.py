import os
import cv2
from ultralytics import YOLO

# 1. Модель # yolo11_dog_detector - назва моделі
model = YOLO("runs/detect/yolo11_dog_detector/weights/best.pt")

# 2. Папки з медіа
input_dir = "/content/drive/MyDrive/Colab Notebooks/input_media"
output_dir = "/content/drive/MyDrive/Colab Notebooks/output_media"
os.makedirs(output_dir, exist_ok=True)

# 3. Підтримувані формати
image_exts = {".jpg", ".jpeg", ".png"}
video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# === 4. Обхід усіх медіафайлів ===
for fname in os.listdir(input_dir):
    input_path = os.path.join(input_dir, fname)
    name, ext = os.path.splitext(fname.lower())

    if ext in image_exts:
        # === Зображення ===
        img = cv2.imread(input_path)
        if img is None:
            print(f"[!] Неможливо прочитати зображення {fname}")
            continue

        results = model(img)
        annotated = results[0].plot()

        output_path = os.path.join(output_dir, f"{name}_detected.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"🖼️ Оброблено зображення: {fname}")

    elif ext in video_exts:
        # === Відео ===
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[!] Неможливо відкрити відео {fname}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            print(f"[!] Невірні розміри для {fname}, пропускаємо.")
            cap.release()
            continue

        out_path = os.path.join(output_dir, f"{name}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frames_written = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            frames_written += 1

        cap.release()
        out.release()

        if frames_written == 0:
            print(f"[!] Жодного кадру не записано у {fname}, видаляємо порожній файл.")
            os.remove(out_path)
        else:
            print(f"🎞️ Оброблено відео: {fname}, кадрів: {frames_written}")

print("✅ Усі медіа оброблено. Результати у 'output_media/'")
