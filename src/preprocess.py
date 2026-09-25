import json
import re
import urllib.request
from collections import Counter

import numpy as np

from .config import DATA_DIR, DATA_PATH, MODEL_DIR, DATA_URL, SEED


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return DATA_PATH

    print("Downloading Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved dataset to: {DATA_PATH}")
    return DATA_PATH


def load_and_clean_text():
    path = download_dataset()
    text = path.read_text(encoding="utf-8", errors="ignore").lower()

    # Remove punctuation and keep alphabetic words plus whitespace.
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def build_vocabulary(tokens):
    # Reserve 0 for unknown/padding.
    counter = Counter(tokens)
    vocab = {"<UNK>": 0}
    for word, _ in counter.most_common():
        vocab[word] = len(vocab)

    id_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, id_to_word, counter


def create_sequences(token_ids, sequence_length, max_sequences=None):
    total = len(token_ids) - sequence_length
    if total <= 0:
        raise ValueError("Not enough tokens for the selected sequence length.")

    if max_sequences is not None:
        total = min(total, max_sequences)

    x = np.empty((total, sequence_length), dtype=np.int32)
    y = np.empty(total, dtype=np.int32)

    for i in range(total):
        x[i] = token_ids[i:i + sequence_length]
        y[i] = token_ids[i + sequence_length]

    return x, y


def prepare_data(sequence_length, max_sequences):
    text = load_and_clean_text()
    tokens = tokenize(text)
    vocab, id_to_word, counter = build_vocabulary(tokens)

    token_ids = np.array([vocab.get(word, 0) for word in tokens], dtype=np.int32)
    x, y = create_sequences(token_ids, sequence_length, max_sequences)

    split = int(len(x) * 0.90)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "word_to_id": vocab,
                "id_to_word": {str(k): v for k, v in id_to_word.items()},
            },
            f,
        )

    stats = {
        "characters_after_cleaning": len(text),
        "total_words": len(tokens),
        "vocabulary_size": len(vocab),
        "sequences_created": len(x),
        "train_sequences": len(x_train),
        "validation_sequences": len(x_val),
        "sequence_length": sequence_length,
        "top_words": counter.most_common(10),
    }

    return x_train, y_train, x_val, y_val, vocab, id_to_word, stats


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, vocab, _, stats = prepare_data(
        sequence_length=20,
        max_sequences=150_000,
    )
    print(json.dumps(stats, indent=2))
