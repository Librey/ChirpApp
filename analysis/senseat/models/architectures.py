"""
senseat/models/architectures.py
================================
Model architectures:
  - SimpleCNN       : baseline (existing)
  - ResNetCNN       : ResNet18-style with BatchNorm (improved)
  - CNN_LSTM        : CNN feature extractor + LSTM temporal modeling
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ─────────────────────────────────────────
# SIMPLE CNN (existing baseline)
# ─────────────────────────────────────────

def build_simple_cnn(input_shape, num_classes=1):
    """
    Original 2-layer CNN baseline.
    num_classes=1  → binary (sigmoid)
    num_classes>1  → multiclass (softmax)
    """
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    loss       = 'binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy'
    out_units  = 1 if num_classes == 1 else num_classes

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(out_units, activation=activation)
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


# ─────────────────────────────────────────
# RESNET-STYLE CNN  (improved)
# ─────────────────────────────────────────

def _residual_block(x, filters, stride=1):
    """Standard ResNet residual block with BatchNorm."""
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_resnet_cnn(input_shape, num_classes=1, dropout_rate=0.4):
    """
    ResNet18-inspired CNN with BatchNorm and residual connections.
    Significantly more robust than SimpleCNN for small datasets.

    num_classes=1  → binary (sigmoid)
    num_classes>1  → multiclass (softmax)
    """
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    loss       = 'binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy'
    out_units  = 1 if num_classes == 1 else num_classes

    inputs = layers.Input(shape=input_shape)

    # Initial conv
    x = layers.Conv2D(32, (3, 3), strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Residual blocks
    x = _residual_block(x, 32)
    x = _residual_block(x, 64, stride=2)
    x = _residual_block(x, 64)
    x = _residual_block(x, 128, stride=2)

    # Global average pooling instead of Flatten — reduces overfitting
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(out_units, activation=activation)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────
# CNN + LSTM  (temporal modeling)
# ─────────────────────────────────────────

def build_cnn_lstm(input_shape, num_classes=1, lstm_units=64, dropout_rate=0.4):
    """
    CNN extracts spatial features from each spectrogram frame.
    LSTM models temporal dependencies across frames (chewing rhythm).

    input_shape: (H, W, 1) — single spectrogram
    Internally reshapes to treat W (time axis) as sequence length.

    num_classes=1  → binary
    num_classes>1  → multiclass
    """
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    loss       = 'binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy'
    out_units  = 1 if num_classes == 1 else num_classes

    H, W, C = input_shape

    inputs = layers.Input(shape=input_shape)

    # CNN feature extractor per time frame
    # Treat each column (time frame) as a sequence element
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)   # pool only frequency axis

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)

    # Reshape: (batch, freq_bins, time_frames, channels) → (batch, time_frames, features)
    new_H = H // 4
    x = layers.Reshape((W, new_H * 64))(x)

    # LSTM temporal modeling
    x = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(out_units, activation=activation)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────

def get_model(model_name, input_shape, num_classes=1, **kwargs):
    """
    Factory function.
    model_name: "simple_cnn" | "resnet" | "cnn_lstm"
    """
    builders = {
        "simple_cnn": build_simple_cnn,
        "resnet":     build_resnet_cnn,
        "cnn_lstm":   build_cnn_lstm,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(builders)}")
    return builders[model_name](input_shape, num_classes=num_classes, **kwargs)
