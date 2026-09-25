import argparse
import json

import numpy as np
import tensorflow as tf

from .config import (
    DEFAULT_GENERATION_WORDS,
    DEFAULT_TEMPERATURE,
    MODEL_DIR,
    OUTPUT_DIR,
    SEQUENCE_LENGTH,
)


def load_resources():
    model = tf.keras.models.load_model(MODEL_DIR / "best_model.keras")

    with open(MODEL_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data["word_to_id"]
    id_to_word = {int(k): v for k, v in vocab_data["id_to_word"].items()}

    return model, word_to_id, id_to_word


def sample_next_word(probabilities, temperature):
    probabilities = np.asarray(probabilities).astype("float64")
    temperature = max(float(temperature), 0.1)

    logits = np.log(probabilities + 1e-8) / temperature
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()

    return int(np.random.choice(len(probabilities), p=probabilities))


def generate_text(model, word_to_id, id_to_word, seed, num_words=40, temperature=0.8):
    words = seed.lower().split()

    if not words:
        raise ValueError("Seed text cannot be empty.")

    for _ in range(num_words):
        sequence = words[-SEQUENCE_LENGTH:]
        ids = [word_to_id.get(word, 0) for word in sequence]

        if len(ids) < SEQUENCE_LENGTH:
            ids = [0] * (SEQUENCE_LENGTH - len(ids)) + ids

        probabilities = model.predict(
            np.array([ids], dtype=np.int32),
            verbose=0,
        )[0]

        next_id = sample_next_word(probabilities, temperature)
        words.append(id_to_word.get(next_id, "<UNK>"))

    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default="to be or not to be",
        help="Seed text for generation.",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=DEFAULT_GENERATION_WORDS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    args = parser.parse_args()

    model, word_to_id, id_to_word = load_resources()

    seeds = [
        args.seed,
        "the king",
        "love is",
    ]

    results = []
    for seed in seeds:
        text = generate_text(
            model,
            word_to_id,
            id_to_word,
            seed,
            num_words=args.words,
            temperature=args.temperature,
        )
        results.append(f"Seed: {seed}\n{text}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "generated_text.txt"
    output_path.write_text("\n".join(results), encoding="utf-8")

    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
