import tensorflow as tf

from .config import DROPOUT, EMBEDDING_DIM, LEARNING_RATE, LSTM_UNITS


def build_model(vocab_size, sequence_length, lstm_layers=1):
    inputs = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)(inputs)

    for layer_index in range(lstm_layers):
        return_sequences = layer_index < lstm_layers - 1
        x = tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=return_sequences,
        )(x)
        if layer_index == 0:
            x = tf.keras.layers.Dropout(DROPOUT)(x)

    outputs = tf.keras.layers.Dense(
        vocab_size,
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )
    return model
