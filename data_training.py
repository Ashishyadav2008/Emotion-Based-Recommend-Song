import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

is_init = False
label = []
dictionary = {}
c = 0

print("🔍 Scanning .npy files...\n")

for i in os.listdir():
    if i.endswith(".npy") and i != "labels.npy":
        try:
            data = np.load(i, allow_pickle=True)
            print(f"📂 {i} loaded | shape: {data.shape} | dtype: {data.dtype}")

            # Skip non-numeric arrays
            if not np.issubdtype(data.dtype, np.number):
                print(f"⚠️ Skipping {i} (non-numeric dtype: {data.dtype})")
                continue

            # Ensure it's 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # Convert to float32
            data = data.astype("float32")

            if not is_init:
                is_init = True
                X = data
                y = np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)
            else:
                # Check shape consistency
                if data.shape[1] == X.shape[1]:
                    X = np.concatenate((X, data))
                    y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))
                else:
                    print(f"⚠️ Skipping {i} (shape mismatch {data.shape} vs {X.shape})")
                    continue

            label.append(i.split('.')[0])
            dictionary[i.split('.')[0]] = c
            c += 1

        except Exception as e:
            print(f"❌ Error loading {i}: {e}")
            continue

# Convert labels to numeric
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode labels
y = to_categorical(y)

# Shuffle data
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
X = X[cnt]
y = y[cnt]

# Build Model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train Model
print("\n🚀 Starting training...\n")
model.fit(X, y, epochs=50, batch_size=32)

# Save Model
model.save("model.h5")
np.save("labels.npy", np.array(label))

print("\n✅ Training complete! Model saved as 'model.h5' and labels as 'labels.npy'.")
