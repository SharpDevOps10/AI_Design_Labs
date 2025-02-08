import numpy as np
from keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense


def generate_xor_data():
    X = np.array([[int(b) for b in format(i, '04b')] for i in range(16)])
    Y = np.array([[np.bitwise_xor.reduce(x)] for x in X])
    return X, Y


X_train, Y_train = generate_xor_data()

model = Sequential([
    keras.Input(shape=(4,)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=1000, verbose=1)

predictions = model.predict(X_train)
predictions = np.round(predictions).astype(int)

print("\nТаблиця істинності XOR (4 входи):")
for x, y_pred in zip(X_train, predictions):
    print(f"Вхід: {x} -> Прогноз: {y_pred[0]}")
