from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

tk = Tokenizer()
tk.fit_on_texts(X_words)
X_tokens = tk.texts_to_sequences(X_text)

# Vocab size?
vocab_size = len(tk.word_index)
vocab_size


def build_model_nlp():
    model = Sequential([
        layers.Embedding(input_dim=vocab_size+1, input_length=maxlen, output_dim=embedding_size, mask_zero=True),
        layers.Conv1D(10, kernel_size=15, padding='same', activation="relu"),
        layers.Conv1D(10, kernel_size=10, padding='same', activation="relu"),
        layers.Flatten(),
        layers.Dense(30, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(1, activation='relu'),
    ])

    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4), metrics=['mae'])
    return model

model_nlp = build_model_nlp()
model_nlp.summary()
