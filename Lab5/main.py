import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "/content/dogs/dog-breeds"

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
print(f"Кількість зображень у датасеті: {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class_names = dataset.classes
num_classes = len(class_names)
print(f"Класи: {class_names}")


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, f1, f3_in, f3_out, f5_in, f5_out, pool_proj):
        super(InceptionBlock, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, f1, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, f3_in, kernel_size=1)
        self.conv3x3_2 = nn.Conv2d(f3_in, f3_out, kernel_size=3, padding=1)
        self.conv5x5_1 = nn.Conv2d(in_channels, f5_in, kernel_size=1)
        self.conv5x5_2 = nn.Conv2d(f5_in, f5_out, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch2 = torch.relu(self.conv3x3_2(torch.relu(self.conv3x3_1(x))))
        branch3 = torch.relu(self.conv5x5_2(torch.relu(self.conv5x5_1(x))))
        branch4 = self.pool_conv(self.pool(x))
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionV3Custom(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Custom, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionBlock(128, 64, 48, 64, 8, 16, 32)
        self.inception2 = InceptionBlock(176, 128, 64, 128, 16, 32, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(352, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


model = InceptionV3Custom(num_classes).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct_val / total_val)

    print(
        f"Епоха {epoch + 1}: Втрата: {train_losses[-1]:.4f}, Точність: {train_accuracies[-1]:.4f} | Валідація - Втрата: {val_losses[-1]:.4f}, Точність: {val_accuracies[-1]:.4f}")


def plot_training():
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Графік втрат')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Графік точності')

    plt.show()


plot_training()

torch.save(model.state_dict(), "inception_v3.pth")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)


def predict_image(image_path):
    model.load_state_dict(torch.load("inception_v3.pth"))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)[0]
        predicted = torch.argmax(output).item()

    print(f"Передбачена порода: {class_names[predicted]}")

    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.title(f"Клас: {class_names[predicted]}")
    plt.show()


predict_image("/content/drive/MyDrive/Colab Notebooks/dogs/PXL_20250128_174735529.jpg")
