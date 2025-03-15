from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

dataset_folder = "/content/drive/MyDrive/Colab Notebooks/yelp_review_polarity_csv/yelp_review_polarity_csv"

train_df = pd.read_csv(os.path.join(dataset_folder, "train.csv"))
test_df = pd.read_csv(os.path.join(dataset_folder, "test.csv"))

train_df.columns = ["label", "text"]
test_df.columns = ["label", "text"]


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


train_df["text"] = train_df["text"].apply(preprocess_text)
test_df["text"] = test_df["text"].apply(preprocess_text)

vocab_size = 20000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

X_train_seq = tokenizer.texts_to_sequences(train_df["text"])
X_test_seq = tokenizer.texts_to_sequences(test_df["text"])

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding="post", truncating="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding="post", truncating="post")

y_train = train_df["label"].values - 1
y_test = test_df["label"].values - 1

embedding_dim = 128

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(
    X_train_padded, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test_padded, y_test)
)

loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"\nТочність на тестових даних: {accuracy:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.title('Графік Точності Моделі')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.title('Графік Втрат Моделі')
plt.legend()
plt.grid(True)
plt.show()


def predict_sentiment_batch(texts):
    if isinstance(texts, str):
        texts = [texts]

    processed_texts = [preprocess_text(text) for text in texts]
    seqs = tokenizer.texts_to_sequences(processed_texts)
    padded_seqs = pad_sequences(seqs, maxlen=max_length, padding="post")

    predictions = model.predict(padded_seqs)

    results = []
    for i, pred in enumerate(predictions):
        probability = pred[0]
        sentiment = "Позитивний" if probability >= 0.5 else "Негативний"
        results.append({
            "текст": texts[i],
            "емоційне забарвлення": sentiment,
            "ймовірність": round(float(probability), 4)
        })

    for res in results:
        print(f"\nТекст: {res['текст']}")
        print(f"Емоційне забарвлення: {res['емоційне забарвлення']} (ймовірність: {res['ймовірність']})")

    return results


texts = [
    "I love this restaurant! The food is amazing and the service is great.",
    "The worst experience ever. I will never come back!",
    "It was okay, nothing special but not terrible either.",
    "Absolutely fantastic! Highly recommend to everyone!",
    "Terrible food, rude staff, and overpriced. Avoid at all costs.",
    "Great place! Cozy atmosphere and friendly staff.",
    "The room was dirty and smelled awful.",
    "Delicious pizza and super fast delivery!",
    "Waiting time was too long, and the food was cold.",
    "Outstanding customer service and wonderful staff!",
    "I found hair in my soup. Disgusting!",
    "Everything was perfect! Would visit again.",
    "Worst hotel ever. Dirty sheets and rude staff.",
    "I'm impressed! The packaging was neat and the product worked flawlessly.",
    "Never seen such a poor management in my life!"
]

predict_sentiment_batch(texts)
