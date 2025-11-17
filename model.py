"""
7-hidden-layer Keras model builder for pushup classification.

Call build_model(input_dim) to get a compiled model.
"""
from tensorflow.keras import layers, models, optimizers

def build_model(input_dim, n_classes=2, dropout_rate=0):
    """
    input_dim: number of input features (should be 4 * len(IMPORTANT_LMS))
    n_classes: 2 for C/L
    returns: compiled Keras model
    """
    model = models.Sequential()
    # # Layer 1
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Layer 2
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Layer 3
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Layer 4
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Layer 5
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Layer 6
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Layer 7
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    # Output
    if n_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(n_classes, activation="softmax"))
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    optimizer = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model