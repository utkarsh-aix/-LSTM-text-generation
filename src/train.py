import json
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from .config import (
    BATCH_SIZE,
    EPOCHS,
    MAX_SEQUENCES,
    MODEL_DIR,
    OUTPUT_DIR,
    SEED,
    SEQUENCE_LENGTH,
)
from .model import build_model
from .preprocess import prepare_data


def main():
    tf.keras.utils.set_random_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing dataset...")
    x_train, y_train, x_val, y_val, vocab, _, stats = prepare_data(
        sequence_length=SEQUENCE_LENGTH,
        max_sequences=MAX_SEQUENCES,
    )

    print(json.dumps(stats, indent=2))
    print(f"Training samples: {len(x_train):,}")
    print(f"Validation samples: {len(x_val):,}")
    print(f"Vocabulary size: {len(vocab):,}")

    model = build_model(
        vocab_size=len(vocab),
        sequence_length=SEQUENCE_LENGTH,
        lstm_layers=1,
    )
    model.summary()

    checkpoint_path = MODEL_DIR / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    start = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    training_seconds = time.time() - start

    model.save(checkpoint_path)

    history_path = OUTPUT_DIR / "training_history.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Shakespeare LSTM Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(history_path, dpi=150)
    plt.close()

    best_val_loss = min(history.history["val_loss"])
    metadata = {
        **stats,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "training_seconds": round(training_seconds, 2),
        "best_validation_loss": round(float(best_val_loss), 5),
        "model_path": str(checkpoint_path),
    }

    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training time: {training_seconds / 60:.2f} minutes")
    print(f"Model: {checkpoint_path}")
    print(f"History plot: {history_path}")


if __name__ == "__main__":
    main()
