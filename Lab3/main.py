import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw
from keras.src.datasets import mnist

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def display_random_images(images, labels, num_images=25):
    plt.figure(figsize=(10, 10))
    random_indices = np.random.choice(images.shape[0], size=num_images, replace=False)
    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        plot_image(images[idx], labels[idx])
    plt.show()
1

def plot_image(img, label):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(label)])


def load_and_preprocess_data():
    train_data, test_data = mnist.load_data()
    x_train_data_normalized, y_train_data = train_data
    x_test_data_normalized, y_test_data = test_data
    x_train_data_normalized = x_train_data_normalized.astype('float32') / 255
    x_test_data_normalized = x_test_data_normalized.astype('float32') / 255
    y_train_data = keras.utils.to_categorical(y_train_data, 10)
    y_test_data = keras.utils.to_categorical(y_test_data, 10)
    print("Shape:", x_train_data_normalized.shape)
    print("train:", x_train_data_normalized.shape[0])
    print("test:", x_test_data_normalized.shape[0])
    return (x_train_data_normalized, y_train_data), (x_test_data_normalized, y_test_data)


def build_model():
    neural_model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    neural_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return neural_model


def train_model(neural_model, x_train_data, y_train_data, epochs=50, batch_size=100, validation_split=0.1):
    history_data = neural_model.fit(x_train_data, y_train_data, batch_size=batch_size,
                                    epochs=epochs, validation_split=validation_split)
    return history_data


def plot_metrics(history_data):
    plt.plot(history_data.history['loss'], label='Training Loss')
    plt.plot(history_data.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.plot(history_data.history['accuracy'], label='Training Accuracy')
    plt.plot(history_data.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def evaluate_model(neural_model, x_test_data, y_test_data):
    loss, accuracy = neural_model.evaluate(x_test_data, y_test_data, verbose=0)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


def test_model_predictions(neural_model, x_test_data, y_test_data, num_samples=5):
    random_indices = np.random.choice(x_test_data.shape[0], size=num_samples, replace=False)
    for idx in random_indices:
        plot_image(x_test_data[idx], np.argmax(y_test_data[idx]))
        prediction = neural_model.predict(x_test_data[idx:idx + 1])
        plt.title("Predicted Number: {}".format(np.argmax(prediction)))
        plt.show()


def predict_digit():
    img_nn = image_canvas.resize((28, 28)).convert('L')
    img_nn = np.array(img_nn)
    img_nn = img_nn / 255.0
    img_nn = img_nn.reshape(1, 28, 28)
    pred = neural_model.predict(img_nn)
    result_label.config(text=f"Розпізнано: {np.argmax(pred)}")


def clear_canvas():
    draw_canvas.rectangle((0, 0, 280, 280), fill='black')
    canvas.delete("all")


def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
    draw_canvas.ellipse([x1, y1, x2, y2], fill='white', outline='white')


if __name__ == "__main__":
    (x_train_data, y_train_data), (x_test_data, y_test_data) = load_and_preprocess_data()
    display_random_images(x_train_data, y_train_data)
    neural_model = build_model()
    neural_model.summary()
    history_data = train_model(neural_model, x_train_data, y_train_data)
    plot_metrics(history_data)
    evaluate_model(neural_model, x_test_data, y_test_data)
    test_model_predictions(neural_model, x_test_data, y_test_data)
    root = tk.Tk()
    root.title("Розпізнавання")
    canvas = tk.Canvas(root, width=280, height=280, bg='black')
    canvas.grid(row=0, column=0, columnspan=2)
    canvas.bind("<B1-Motion>", paint)
    image_canvas = Image.new('RGB', (280, 280), 'black')
    draw_canvas = ImageDraw.Draw(image_canvas)
    btn_predict = tk.Button(root, text="Розпізнати", command=predict_digit)
    btn_predict.grid(row=1, column=0)
    btn_clear = tk.Button(root, text="Очистити", command=clear_canvas)
    btn_clear.grid(row=1, column=1)
    result_label = tk.Label(root, text="Розпізнано: ")
    result_label.grid(row=2, column=0, columnspan=2)
    root.mainloop()
