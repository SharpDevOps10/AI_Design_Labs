import tensorflow as tf
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

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

X_train_seq = tokenizer.texts_to_sequences(train_df["text"])
X_test_seq = tokenizer.texts_to_sequences(test_df["text"])

X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_length, padding="post",
                                                               truncating="post")
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_length, padding="post",
                                                              truncating="post")

y_train = train_df["label"].values - 1
y_test = test_df["label"].values - 1


class VanillaLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(VanillaLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [self.units, self.units]  # h_t, c_t
        self.output_size = self.units  # розмір h_t

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Основна матриця ваг для всіх чотирьох вентилів
        self.W = self.add_weight(
            shape=(input_dim + self.units, self.units * 4),
            initializer='glorot_uniform',
            trainable=True,
            name="W"
        )

        # Ініціалізація bias з урахуванням forget_bias=1.0
        forget_bias_value = 1.0
        b_i = tf.zeros(self.units)
        b_f = tf.ones(self.units) * forget_bias_value  # Forget gate bias = 1.0
        b_c = tf.zeros(self.units)
        b_o = tf.zeros(self.units)
        b_init = tf.concat([b_i, b_f, b_c, b_o], axis=0)

        self.b = self.add_weight(
            shape=(self.units * 4,),
            initializer=tf.keras.initializers.Constant(b_init),
            trainable=True,
            name="b"
        )

        # Peephole connections
        self.peephole_i = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name="peephole_i"
        )
        self.peephole_f = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name="peephole_f"
        )
        self.peephole_o = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name="peephole_o"
        )

    def call(self, x, states):
        h_prev, c_prev = states

        # Комбінований вхід
        combined = tf.concat([x, h_prev], axis=-1)

        # Лінійна трансформація для всіх вентилів
        z = tf.matmul(combined, self.W) + self.b

        # Розділяємо на вентилі
        i, f, c_hat, o = tf.split(z, num_or_size_splits=4, axis=1)

        # Peephole connections додаються до вхідного, forget і вихідного вентилів
        i += self.peephole_i * c_prev
        f += self.peephole_f * c_prev
        o += self.peephole_o * c_prev

        # Активації
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        c_hat = tf.tanh(c_hat)

        # Оновлення стану комірки
        c = f * c_prev + i * c_hat

        # Оновлення вихідного стану
        h = o * tf.tanh(c)

        return h, [h, c]


batch_size = 512

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

embedding_dim = 128
projection_dim = 64
lstm_units = 128

inputs = tf.keras.Input(shape=(max_length,))
x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

vanilla_lstm_cell = VanillaLSTMCell(lstm_units)
rnn_output = tf.keras.layers.RNN(vanilla_lstm_cell, return_sequences=True)(x)

# Використовуємо останній hidden state
x = rnn_output[:, -1, :]

# Класифікаційна частина
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Модель
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset
)

loss, accuracy = model.evaluate(test_dataset)
print(f"\nТочність на тестових даних: {accuracy:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.title('Графік точності моделі')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.title('Графік втрат моделі')
plt.legend()
plt.grid(True)
plt.show()


def predict_sentiment_batch(texts):
    if isinstance(texts, str):
        texts = [texts]

    processed_texts = [preprocess_text(text) for text in texts]
    seqs = tokenizer.texts_to_sequences(processed_texts)
    padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length, padding="post")

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
