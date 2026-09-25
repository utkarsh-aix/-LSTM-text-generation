import json
import time

import numpy as np
import tensorflow as tf

from .config import MODEL_DIR, OUTPUT_DIR, SEED
from .model import build_model
from .preprocess import prepare_data
from .generate import generate_text


def run_experiment(name, sequence_length, lstm_layers, max_sequences=75_000, epochs=2):
    print(f"\n=== {name} ===")
    x_train, y_train, x_val, y_val, vocab, id_to_word, stats = prepare_data(
        sequence_length=sequence_length,
        max_sequences=max_sequences,
    )

    model = build_model(
        vocab_size=len(vocab),
        sequence_length=sequence_length,
        lstm_layers=lstm_layers,
    )

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )
    elapsed = time.time() - start

    result = {
        "name": name,
        "sequence_length": sequence_length,
        "lstm_layers": lstm_layers,
        "best_train_loss": float(min(history.history["loss"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "training_seconds": round(elapsed, 2),
        "generated_text": generate_text(
            model,
            vocab,
            id_to_word,
            "to be or not to be",
            num_words=30,
            temperature=0.8,
        ),
        "data": stats,
    }
    return result


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        run_experiment(
            "baseline",
            sequence_length=20,
            lstm_layers=1,
        ),
        run_experiment(
            "deeper_longer",
            sequence_length=40,
            lstm_layers=2,
        ),
    ]

    output_path = OUTPUT_DIR / "experiment_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for result in results:
        print(f"\n{result['name']}")
        print(f"Validation loss: {result['best_val_loss']:.4f}")
        print(f"Training time: {result['training_seconds'] / 60:.2f} minutes")
        print(result["generated_text"])


if __name__ == "__main__":
    main()
