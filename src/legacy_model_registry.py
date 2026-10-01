"""Model registry for legacy-compatible and current KIER experiments.

This module keeps model definitions separate from experiment orchestration so
the same aggregate/cluster experiment can swap ML and DL models by name.

The legacy names reproduce the previous repository's ML/DL model families
without importing or modifying that repository. The modern DL names are the
models designed in this project and can be plugged into the same aggregate
experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ML_MODEL_NAMES = ("CatBoost", "DecisionTree", "LightGBM", "RandomForest", "XGBoost")
DL_MODEL_NAMES = (
    "Legacy1DCNNLSTM",
    "Legacy1DCNNSeq2Seq",
    "1D_CNN_LSTM",
    "GRU",
    "Transformer",
    "TCN",
    "RetNet",
)
MODEL_ALIASES = {
    "1D-CNN_LSTM": "Legacy1DCNNLSTM",
    "1D-CNN_Seq2Seq": "Legacy1DCNNSeq2Seq",
    "CNNLSTM": "1D_CNN_LSTM",
}

tf = None
keras = None
EarlyStopping = None
ReduceLROnPlateau = None


@dataclass(frozen=True)
class DLTrainConfig:
    epochs: int = 200
    batch_size: int = 128
    patience: int = 20
    learning_rate: float = 1e-3
    loss: str = "huber"


def ensure_tf() -> None:
    global tf, keras, EarlyStopping, ReduceLROnPlateau
    if tf is not None:
        return
    import tensorflow as _tf
    from tensorflow import keras as _keras
    from keras.callbacks import EarlyStopping as _EarlyStopping
    from keras.callbacks import ReduceLROnPlateau as _ReduceLROnPlateau

    tf = _tf
    keras = _keras
    EarlyStopping = _EarlyStopping
    ReduceLROnPlateau = _ReduceLROnPlateau


def build_ml_model(name: str, seed: int = 42) -> Any:
    """Return the legacy ML estimator for a model name."""
    if name == "CatBoost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=500,
            max_ctr_complexity=6,
            random_seed=10,
            od_type="Iter",
            od_wait=25,
            verbose=0,
            depth=5,
            learning_rate=0.03,
        )
    if name == "DecisionTree":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(max_depth=8, random_state=seed)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=10000, learning_rate=0.01, verbose=-1, random_state=seed)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            max_depth=8,
            min_samples_leaf=8,
            min_samples_split=8,
            n_estimators=200,
            random_state=seed,
            n_jobs=1,
        )
    if name == "XGBoost":
        import xgboost as xgb

        return xgb.XGBRegressor(n_estimators=1000, random_state=seed, tree_method="hist", verbosity=0)
    raise ValueError(f"Unsupported ML model: {name}")


def normalize_model_name(name: str) -> str:
    """Return the registry's canonical model name for legacy aliases."""
    return MODEL_ALIASES.get(name, name)


def _legacy_1dcnn_lstm(n_features: int, seq_len: int):
    ensure_tf()
    activation = "swish"
    inp = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.Conv1D(512, 1, activation=activation)(inp)
    x = keras.layers.MaxPool1D(pool_size=2, strides=1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(1024, 1, activation=activation)(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(1024, activation=activation, dropout=0.15, return_sequences=True)(x)
    x = keras.layers.LSTM(512, activation=activation, dropout=0.15, return_sequences=True)(x)
    x = keras.layers.LSTM(256, activation=activation, dropout=0.15, return_sequences=True)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation=activation)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out, name="Legacy1DCNNLSTM")


def _legacy_1dcnn_seq2seq(n_features: int, seq_len: int):
    ensure_tf()
    inp = keras.Input(shape=(seq_len, n_features))
    conv1 = keras.layers.Conv1D(1024, 1, activation="swish")(inp)
    pool1 = keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(conv1)
    bat1 = keras.layers.BatchNormalization()(pool1)
    conv2 = keras.layers.Conv1D(512, 1, activation="swish")(bat1)
    pool2 = keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(conv2)
    bat2 = keras.layers.BatchNormalization()(pool2)
    enc1 = keras.layers.LSTM(256, return_sequences=True, activation="swish")(bat2)
    enc2 = keras.layers.LSTM(512, return_sequences=True, activation="swish")(bat1)
    enc3, state_h, state_c = keras.layers.LSTM(
        1024, return_state=True, return_sequences=True, activation="swish"
    )(enc2)
    dec1 = keras.layers.LSTM(1024, return_sequences=True, activation="swish")(
        enc3, initial_state=[state_h, state_c]
    )
    dec2 = keras.layers.LSTM(512, return_sequences=True, activation="swish")(dec1)
    dec3 = keras.layers.LSTM(256, return_sequences=True, activation="swish")(dec2)
    flat = keras.layers.Flatten()(dec3)
    out = keras.layers.Dense(1)(flat)
    return keras.Model(inp, out, name="Legacy1DCNNSeq2Seq")


def _modern_1dcnn_lstm(n_features: int, seq_len: int):
    ensure_tf()
    inp = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.Conv1D(64, 3, padding="causal", activation="swish")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(128, 3, padding="causal", activation="swish")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(128, return_sequences=True, dropout=0.15)(x)
    x = keras.layers.LSTM(64, return_sequences=False, dropout=0.15)(x)
    x = keras.layers.Dense(32, activation="swish")(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out, name="1D_CNN_LSTM")


def _gru(n_features: int, seq_len: int):
    ensure_tf()
    inp = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.GRU(128, return_sequences=True, dropout=0.15)(inp)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GRU(64, return_sequences=False, dropout=0.15)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out, name="GRU")


def _pos_encoding(seq_len: int, d_model: int):
    ensure_tf()
    import numpy as np

    positions = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[None], dtype=tf.float32)


def _transformer(n_features: int, seq_len: int):
    ensure_tf()
    d_model = 64
    inp = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.Dense(d_model)(inp)
    x = x + _pos_encoding(seq_len, d_model)
    x = keras.layers.Dropout(0.1)(x)
    for _ in range(2):
        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = keras.layers.Dense(128, activation="relu")(x)
        ffn = keras.layers.Dense(d_model)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + keras.layers.Dropout(0.1)(ffn))
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out, name="Transformer")


def _tcn_block(x, filters: int, dilation: int):
    ensure_tf()
    residual = x
    for _ in range(2):
        x = keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(0.1)(x)
    if residual.shape[-1] != filters:
        residual = keras.layers.Conv1D(filters, 1)(residual)
    return keras.layers.Add()([x, residual])


def _tcn(n_features: int, seq_len: int):
    ensure_tf()
    inp = keras.Input(shape=(seq_len, n_features))
    x = inp
    for dilation in [1, 2, 4, 8, 16]:
        x = _tcn_block(x, 64, dilation)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out, name="TCN")


def _retnet(n_features: int, seq_len: int):
    ensure_tf()
    d_model = 64

    class MultiScaleRetention(keras.layers.Layer):
        """Compact RetNet-style retention layer used by this project."""

        def __init__(self, d_model: int, num_heads: int, **kwargs):
            super().__init__(**kwargs)
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads")
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.d_model = d_model
            self.gammas = [1.0 - 2.0 ** (-5 - h) for h in range(num_heads)]
            self.w_q = keras.layers.Dense(d_model, use_bias=False)
            self.w_k = keras.layers.Dense(d_model, use_bias=False)
            self.w_v = keras.layers.Dense(d_model, use_bias=False)
            self.w_g = keras.layers.Dense(d_model)
            self.w_o = keras.layers.Dense(d_model)
            self.ln = keras.layers.LayerNormalization()

        def call(self, x, training=None):
            batch = tf.shape(x)[0]
            steps = tf.shape(x)[1]
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
            gate = tf.nn.swish(self.w_g(x))

            def split_heads(z):
                z = tf.reshape(z, [batch, steps, self.num_heads, self.head_dim])
                return tf.transpose(z, [0, 2, 1, 3])

            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            idx = tf.cast(tf.range(steps), tf.float32)
            diff = idx[:, tf.newaxis] - idx[tf.newaxis, :]
            causal = tf.cast(diff >= 0, tf.float32)
            scale = tf.cast(self.head_dim, tf.float32) ** 0.5

            heads_out = []
            for head in range(self.num_heads):
                decay = tf.pow(self.gammas[head], tf.maximum(diff, 0.0)) * causal
                scores = tf.matmul(q[:, head], k[:, head], transpose_b=True) / scale
                heads_out.append(tf.matmul(scores * decay[tf.newaxis], v[:, head]))

            retained = tf.stack(heads_out, axis=2)
            retained = tf.reshape(retained, [batch, steps, self.d_model])
            return self.w_o(self.ln(retained) * gate)

    inp = keras.Input(shape=(seq_len, n_features))
    x = keras.layers.Dense(d_model)(inp)
    x = keras.layers.Dropout(0.1)(x)
    for _ in range(2):
        residual = x
        ret = MultiScaleRetention(d_model=d_model, num_heads=4)(x)
        x = keras.layers.LayerNormalization()(residual + keras.layers.Dropout(0.1)(ret))
        residual = x
        ffn = keras.layers.Dense(128, activation="gelu")(x)
        ffn = keras.layers.Dense(d_model)(ffn)
        x = keras.layers.LayerNormalization()(residual + keras.layers.Dropout(0.1)(ffn))
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="gelu")(x)
    out = keras.layers.Dense(1)(x)
    return keras.Model(inp, out, name="RetNet")


DL_BUILDERS: dict[str, Callable[[int, int], Any]] = {
    "Legacy1DCNNLSTM": _legacy_1dcnn_lstm,
    "Legacy1DCNNSeq2Seq": _legacy_1dcnn_seq2seq,
    "1D_CNN_LSTM": _modern_1dcnn_lstm,
    "GRU": _gru,
    "Transformer": _transformer,
    "TCN": _tcn,
    "RetNet": _retnet,
}


def build_dl_model(name: str, n_features: int, seq_len: int):
    name = normalize_model_name(name)
    if name not in DL_BUILDERS:
        raise ValueError(f"Unsupported DL model: {name}")
    return DL_BUILDERS[name](n_features, seq_len)


def compile_dl_model(model: Any, config: DLTrainConfig) -> None:
    ensure_tf()
    loss = keras.losses.Huber() if config.loss == "huber" else "mse"
    optimizer = keras.optimizers.legacy.Adam(learning_rate=config.learning_rate, clipnorm=1.0)
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])


def dl_callbacks(config: DLTrainConfig) -> list[Any]:
    ensure_tf()
    return [
        EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]
