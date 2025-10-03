import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialization
is_init = False
label = []
dictionary = {}
c = 0

# Load and concatenate data
for i in os.listdir():
    if i.endswith(".npy") and i != "labels.npy":
        data = np.load(i)

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        size = data.shape[0]
        class_name = i.split('.')[0]
        class_labels = np.array([class_name] * size).reshape(-1, 1)

        if not is_init:
            X = data
            y = class_labels
            is_init = True
        else:
            X = np.concatenate((X, data), axis=0)
            y = np.concatenate((y, class_labels), axis=0)

        label.append(class_name)
        dictionary[class_name] = c
        c += 1

# Convert class names in `y` to integer labels using the dictionary
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

y = np.array(y, dtype="int32")
y = to_categorical(y)  # One-hot encoding

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# Build model
ip = Input(shape=(X.shape[1],))  # Input shape based on feature count

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train model
model.fit(X, y, epochs=50)

# Save model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))

print("Training complete. Model and labels saved.")
