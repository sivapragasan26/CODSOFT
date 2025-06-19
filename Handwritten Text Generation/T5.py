import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

# Step 1: Sample text dataset (replace with handwriting transcription)
text = """This is a sample text to simulate handwritten text generation using a character-level RNN model. You can replace this with actual handwriting transcription data."""

# Convert to lowercase
text = text.lower()

# Step 2: Create character-to-index mappings
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"Unique characters: {vocab_size}")

# Step 3: Prepare sequences
seq_length = 40
step = 3
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print("Total sequences:", len(sequences))

# Step 4: Vectorize sequences
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=bool)
y = np.zeros((len(sequences), vocab_size), dtype=bool)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Step 5: Build RNN model
model = Sequential([
    LSTM(128, input_shape=(seq_length, vocab_size)),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Step 6: Train the model
model.fit(X, y, batch_size=128, epochs=20)

# Step 7: Text Generation Function
def generate_text(seed, length=300):
    seed = seed.lower()
    if len(seed) < seq_length:
        seed = (" " * (seq_length - len(seed))) + seed
    seed = seed[-seq_length:]

    generated = seed
    for _ in range(length):
        input_seq = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(seed):
            if char in char_to_idx:
                input_seq[0, t, char_to_idx[char]] = 1
        pred = model.predict(input_seq, verbose=0)[0]
        next_index = np.random.choice(len(pred), p=pred)
        next_char = idx_to_char[next_index]
        generated += next_char
        seed = seed[1:] + next_char

    return generated

# Step 8: Try generating text
seed_text = "this is a sample"
print("\nGenerated text:\n")
print(generate_text(seed_text, length=300))
