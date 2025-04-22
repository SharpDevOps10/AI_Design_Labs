from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/Colab Notebooks/yolo11.yaml")

model.train(
    data="/content/drive/MyDrive/Colab Notebooks/dogs.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    name="yolo11_dog_detector",
    device=0
)
