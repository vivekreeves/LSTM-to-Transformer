import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_doc(filename):
    """Load text from a file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find input file: {filename}")

    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def clean_lines(doc_text):
    """Normalize text into a list of non-empty lines."""
    lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The document has no usable text lines.")
    return lines


def build_training_data(doc_text, max_seq_len=None):
    """Convert text into padded training sequences and labels."""
    lines = clean_lines(doc_text)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    sequences = tokenizer.texts_to_sequences(lines)
    sequences = [seq for seq in sequences if seq]
    if not sequences:
        raise ValueError("No tokenized sequences were created from the input text.")

    max_seq_len = max_seq_len or max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, maxlen=max_seq_len, padding="pre")

    X = padded[:, :-1]
    y = padded[:, -1]
    seq_length = X.shape[1]
    vocab_size = len(tokenizer.word_index) + 1
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    return X, y, tokenizer, seq_length, vocab_size


def define_model(vocab_size, seq_length, embedding_dim=128, rnn_units=256, dropout=0.2):
    """Build the LSTM language model."""
    inputs = tf.keras.Input(shape=(seq_length,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = tf.keras.layers.LSTM(rnn_units)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics=["accuracy"],
    )
    return model


def train_model(model, X, y, output_dir, epochs=100, batch_size=128):
    """Train the model and checkpoint the best weights."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "model-best.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    return model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )


def save_model_artifacts(model, tokenizer, output_dir):
    """Persist the trained model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model-best.keras")
    tokenizer_path = os.path.join(output_dir, "tokenizer.pkl")

    model.save(model_path)
    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle)

    return model_path, tokenizer_path


def top_k_top_p_filter(probs, top_k=50, top_p=0.9):
    """Apply top-k and top-p filtering to probability scores."""
    probs = np.asarray(probs, dtype=np.float64).copy()
    probs[0] = 0.0

    if top_k is not None and 0 < top_k < probs.size:
        idx = np.argpartition(-probs, top_k)[:top_k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[idx] = True
        probs[~mask] = 0.0

    if 0 < top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        cutoff_index = np.argmax(cumulative > top_p)
        if cumulative[cutoff_index] > top_p:
            keep = sorted_idx[: cutoff_index + 1]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs[~mask] = 0.0

    total = probs.sum()
    if total <= 0:
        return probs
    return probs / total


def apply_temperature(probs, temperature=1.1):
    """Adjust the next-token distribution for sampling diversity."""
    if temperature and temperature != 1.0:
        logits = np.log(np.maximum(probs, 1e-12)) / float(temperature)
        logits -= logits.max()
        probs = np.exp(logits)
        total = probs.sum()
        if total <= 0:
            return probs
        probs /= total
    return probs


def penalize_repetition(probs, recent_ids, freq_penalty=0.6):
    """Reduce the chance of repeating recent tokens."""
    if not recent_ids or freq_penalty <= 0:
        return probs

    counts = {}
    for token_id in recent_ids:
        counts[token_id] = counts.get(token_id, 0) + 1

    for token_id, count in counts.items():
        probs[token_id] /= (1.0 + freq_penalty * count)

    total = probs.sum()
    if total <= 0:
        return probs
    return probs / total


def violates_no_repeat_ngram(candidate_id, history_ids, n=3):
    """Check whether a candidate token would repeat a recent n-gram."""
    if n <= 1 or len(history_ids) < n - 1:
        return False

    candidate_ngram = history_ids[-(n - 1) :] + [candidate_id]
    for idx in range(len(history_ids) - (n - 1)):
        if history_ids[idx : idx + n] == candidate_ngram:
            return True
    return False


def choose_next_token(probs, history_ids, no_repeat_ngram=3):
    """Sample a token while avoiding repeated n-grams when possible."""
    if no_repeat_ngram and no_repeat_ngram >= 2 and len(history_ids) >= no_repeat_ngram - 1:
        for _ in range(5):
            candidate = int(np.random.choice(len(probs), p=probs))
            if not violates_no_repeat_ngram(candidate, history_ids, n=no_repeat_ngram):
                return candidate
            probs[candidate] = 0.0
            total = probs.sum()
            if total <= 0:
                break
            probs = probs / total
        return int(np.argmax(probs))

    return int(np.random.choice(len(probs), p=probs))


def generate_text(model, tokenizer, seed_text, n_words=50, seq_length=None, temperature=1.1, top_k=50, top_p=0.9, freq_penalty=0.6, no_repeat_ngram=3, recent_window=20):
    """Generate text from a trained language model."""
    if seq_length is None:
        try:
            seq_length = model.input_shape[1]
        except Exception as exc:  # pragma: no cover
            raise ValueError("The model sequence length is unavailable.") from exc

    index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    current_text = seed_text.strip()
    result_words = []
    history_ids = []

    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([current_text])[0]
        if not encoded:
            break

        padded = pad_sequences([encoded], maxlen=seq_length, truncating="pre")
        probs = model.predict(padded, verbose=0)[0]
        probs = penalize_repetition(probs, history_ids[-recent_window:], freq_penalty=freq_penalty)
        probs = apply_temperature(probs, temperature=temperature)
        probs = top_k_top_p_filter(probs, top_k=top_k, top_p=top_p)

        next_id = choose_next_token(probs, history_ids, no_repeat_ngram=no_repeat_ngram)
        word = index_word.get(next_id, "")
        if not word or word in {"<unk>", "unk"}:
            break

        result_words.append(word)
        history_ids.append(next_id)
        current_text += " " + word

    return " ".join(result_words)
