"""
Common_Model_DL.py
DL 모델 학습·예측·평가 공통 모듈
  - 1D-CNN LSTM / 1D-CNN Seq2Seq
  - TCN (Temporal Convolutional Network)
  - Transformer Encoder
  - RetNet (Retentive Network)

History
  2024-04-05  Created
  2026-03-30  Refactored - import 정리, 중복 함수 제거 (Common_Model_ML에서 import)
  2026-04-15  TCN / Transformer / RetNet 모델 추가
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold

from core.model_ml import (
    model_sk_metrics,
    model_visualization,
    _aggregate_kfold_scores,
)


# =========================================================================== #
#  Data Split (DL 전용: 시퀀스 윈도우 생성)
# =========================================================================== #

def _build_sequences(X, Y, seq_len: int):
    """슬라이딩 윈도우로 (X_seq, Y_seq) 배열 생성."""
    x_data, y_data = [], []
    for i in range(len(X) - seq_len + 1):
        x_data.append(X.iloc[i:i + seq_len].values)
        y_data.append(Y.iloc[i + seq_len - 1].values)
    return np.array(x_data), np.array(y_data)


def split_build_dataset(df: pd.DataFrame, test_ratio: float,
                        target_col: str, seq_len: int):
    """Train/Test 분할 후 시퀀스 윈도우 생성."""
    train, test = train_test_split(df, test_size=test_ratio, shuffle=False)
    train_x, train_y = train.drop(columns=[target_col]), train[[target_col]]
    test_x, test_y = test.drop(columns=[target_col]), test[[target_col]]

    train_X, train_Y = _build_sequences(train_x, train_y, seq_len)
    test_X, test_Y = _build_sequences(test_x, test_y, seq_len)
    return train_X, test_X, train_Y, test_Y


# =========================================================================== #
#  Model Builders
# =========================================================================== #

def build_1dcnn_lstm(n_features: int, seq_len: int = 3,
                     activation: str = "swish"):
    """KIER M02 — 1D-CNN + LSTM 모델."""
    inp = tf.keras.layers.Input(shape=(seq_len, n_features))

    x = tf.keras.layers.Conv1D(512, 1, activation=activation)(inp)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(1024, 1, activation=activation)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LSTM(1024, activation=activation, dropout=0.15, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(512,  activation=activation, dropout=0.15, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(256,  activation=activation, dropout=0.15, return_sequences=True)(x)

    for units in [256, 128, 64]:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units, activation=activation)(x)

    out = tf.keras.layers.Dense(1)(tf.keras.layers.BatchNormalization()(x))
    return "1D-CNN_LSTM", tf.keras.models.Model(inp, out)


def build_1dcnn_seq2seq(input_shape, activation: str = "swish"):
    """KIER M02 — 1D-CNN + Seq2Seq 모델."""
    inp = tf.keras.layers.Input(shape=input_shape)

    c1 = tf.keras.layers.Conv1D(1024, 1, activation=activation)(inp)
    p1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(c1)
    b1 = tf.keras.layers.BatchNormalization()(p1)
    c2 = tf.keras.layers.Conv1D(512, 1, activation=activation)(b1)
    p2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(c2)
    b2 = tf.keras.layers.BatchNormalization()(p2)

    # Encoder
    enc1 = tf.keras.layers.LSTM(256, return_sequences=True, activation=activation)(b2)
    enc2 = tf.keras.layers.LSTM(512, return_sequences=True, activation=activation)(b1)
    enc3_out, state_h, state_c = tf.keras.layers.LSTM(
        1024, return_state=True, return_sequences=True, activation=activation)(enc2)

    # Decoder
    dec1 = tf.keras.layers.LSTM(1024, return_sequences=True, activation=activation)(
        enc3_out, initial_state=[state_h, state_c])
    dec2 = tf.keras.layers.LSTM(512, return_sequences=True, activation=activation)(dec1)
    dec3 = tf.keras.layers.LSTM(256, return_sequences=True, activation=activation)(dec2)

    flat = tf.keras.layers.Flatten()(dec3)
    out = tf.keras.layers.Dense(1)(flat)
    return "1D-CNN_Seq2Seq", tf.keras.models.Model(inp, out)


# =========================================================================== #
#  Model Builders — GRU
# =========================================================================== #

def build_gru(n_features: int, seq_len: int,
              units: list | None = None,
              dropout_rate: float = 0.15,
              activation: str = "tanh"):
    """
    KIER M02 — Stacked GRU 예측 모델.

    Gated Recurrent Unit: LSTM 대비 파라미터 수 ~33% 적고 학습 속도 빠름.
    마지막 GRU 층은 return_sequences=False → (batch, units[-1]) 직접 출력.
    """
    if units is None:
        units = [256, 128, 64]

    inp = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = inp
    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        x = tf.keras.layers.GRU(
            u, return_sequences=return_seq,
            dropout=dropout_rate, activation=activation,
        )(x)
        if return_seq:
            x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    return "GRU", tf.keras.models.Model(inp, out)


# =========================================================================== #
#  Model Builders — TCN
# =========================================================================== #

def _tcn_block(x, filters: int, kernel_size: int, dilation_rate: int,
               dropout_rate: float, activation: str):
    """
    TCN 잔차 블록.
    2× (dilated causal Conv1D → LayerNorm → Activation → Dropout) + residual 투영.
    """
    residual = x

    for _ in range(2):
        x = tf.keras.layers.Conv1D(
            filters, kernel_size,
            padding="causal", dilation_rate=dilation_rate,
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # 차원 불일치 시 residual 투영
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(filters, 1)(residual)

    return tf.keras.layers.Add()([x, residual])


def build_tcn(n_features: int, seq_len: int,
              nb_filters: int = 64,
              kernel_size: int = 3,
              dilations: list | None = None,
              dropout_rate: float = 0.1,
              activation: str = "relu"):
    """
    KIER M02 — TCN (Temporal Convolutional Network) 모델.

    dilated causal convolution + 잔차 연결로 장기 의존성을 포착.
    수용 영역(receptive field) = 2 × (kernel_size-1) × Σ(dilations).
    """
    if dilations is None:
        dilations = [1, 2, 4, 8]

    inp = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = inp
    for d in dilations:
        x = _tcn_block(x, nb_filters, kernel_size, d, dropout_rate, activation)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation=activation)(x)
    out = tf.keras.layers.Dense(1)(x)
    return "TCN", tf.keras.models.Model(inp, out)


# =========================================================================== #
#  Model Builders — Transformer
# =========================================================================== #

def _transformer_encoder_block(x, d_model: int, num_heads: int,
                                ff_dim: int, dropout_rate: float):
    """Transformer Encoder 블록: MHA → Add&Norm → FFN → Add&Norm."""
    # Multi-Head Self-Attention
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
    )(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Position-wise Feed-Forward
    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x


def build_transformer(n_features: int, seq_len: int,
                      d_model: int = 128,
                      num_heads: int = 4,
                      ff_dim: int = 256,
                      num_layers: int = 2,
                      dropout_rate: float = 0.1):
    """
    KIER M02 — Transformer Encoder 기반 예측 모델.

    Sinusoidal positional encoding + Multi-Head Self-Attention + FFN.
    """
    inp = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.Dense(d_model)(inp)

    # Sinusoidal positional encoding
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_enc = tf.cast(angles[np.newaxis], dtype=tf.float32)  # (1, seq_len, d_model)
    x = x + pos_enc
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for _ in range(num_layers):
        x = _transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    return "Transformer", tf.keras.models.Model(inp, out)


# =========================================================================== #
#  Model Builders — RetNet
# =========================================================================== #

class _MultiScaleRetention(tf.keras.layers.Layer):
    """
    Multi-Scale Retention (병렬 모드).

    RetNet 핵심 블록 — Sun et al. 2023 "Retentive Network" §3.
    decay γ_h = 1 - 2^(-5-h),  h = 0 … num_heads-1
    D[m,n] = γ^(m-n) if m ≥ n else 0  (인과 감쇠 마스크)
    Ret_h(X) = (Q_h K_h^T ⊙ D / √d_h) V_h
    출력 = W_O( LayerNorm(concat(Ret_h)) ⊙ swish(W_G X) )
    """

    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.d_model   = d_model
        self.gammas    = [1.0 - 2.0 ** (-5 - h) for h in range(num_heads)]

        self.W_Q = tf.keras.layers.Dense(d_model, use_bias=False)
        self.W_K = tf.keras.layers.Dense(d_model, use_bias=False)
        self.W_V = tf.keras.layers.Dense(d_model, use_bias=False)
        self.W_G = tf.keras.layers.Dense(d_model)
        self.W_O = tf.keras.layers.Dense(d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=None):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        Q = self.W_Q(x)                                         # (B, T, d_model)
        K = self.W_K(x)
        V = self.W_V(x)
        G = tf.nn.swish(self.W_G(x))

        # (B, T, d_model) → (B, H, T, d_h)
        def split_heads(z):
            z = tf.reshape(z, [B, T, self.num_heads, self.head_dim])
            return tf.transpose(z, [0, 2, 1, 3])

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 인과 감쇠 마스크 (공통)
        idx  = tf.cast(tf.range(T), tf.float32)
        diff = idx[:, tf.newaxis] - idx[tf.newaxis, :]          # (T, T)
        causal = tf.cast(diff >= 0, tf.float32)

        scale = tf.cast(self.head_dim, tf.float32) ** 0.5
        heads_out = []
        for h in range(self.num_heads):
            D = tf.pow(self.gammas[h], tf.maximum(diff, 0.0)) * causal   # (T, T)
            scores = tf.matmul(Q[:, h], K[:, h], transpose_b=True) / scale  # (B,T,T)
            ret_h  = tf.matmul(scores * D[tf.newaxis], V[:, h])           # (B,T,d_h)
            heads_out.append(ret_h)

        # (B, T, H, d_h) → (B, T, d_model)
        ret = tf.stack(heads_out, axis=2)
        ret = tf.reshape(ret, [B, T, self.d_model])

        ret = self.layer_norm(ret) * G
        return self.W_O(ret)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


def build_retnet(n_features: int, seq_len: int,
                 d_model: int = 128,
                 num_heads: int = 4,
                 ff_dim: int = 256,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1):
    """
    KIER M02 — RetNet (Retentive Network) 예측 모델.

    Multi-Scale Retention + FFN, num_layers 회 적층.
    Transformer 대비 O(T) 추론 복잡도 (병렬 모드 학습, 순환 모드 추론).
    """
    inp = tf.keras.layers.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.Dense(d_model)(inp)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for _ in range(num_layers):
        # Retention 블록
        residual = x
        ret = _MultiScaleRetention(d_model, num_heads)(x)
        ret = tf.keras.layers.Dropout(dropout_rate)(ret)
        x = tf.keras.layers.LayerNormalization()(residual + ret)

        # FFN 블록
        residual = x
        ffn = tf.keras.layers.Dense(ff_dim, activation="gelu")(x)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        x = tf.keras.layers.LayerNormalization()(residual + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="gelu")(x)
    out = tf.keras.layers.Dense(1)(x)
    return "RetNet", tf.keras.models.Model(inp, out)


# =========================================================================== #
#  DL: Predict
# =========================================================================== #

def model_dl_predict(train_X, train_Y, test_X, test_Y,
                     model, epochs: int = 500, batch_size: int = 32,
                     patience: int = 15, lr: float = 3e-4):
    """
    DL 모델 학습·예측.

    Returns : (y_actual, y_preds, elapsed_sec)
    """
    t0 = time.time()
    es = EarlyStopping(monitor="loss", patience=patience)
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adamax(learning_rate=lr, clipnorm=1.0),
        metrics=["mae"],
    )
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, callbacks=[es])
    elapsed = time.time() - t0

    preds = model.predict(test_X).reshape(-1, 1)
    actual = test_Y.reshape(-1, 1)
    return actual, preds, elapsed


# =========================================================================== #
#  DL: Single Analysis
# =========================================================================== #

def model_dl_analysis_single(df: pd.DataFrame, model,
                             test_ratio: float, target_col: str,
                             seq_len: int):
    train_X, test_X, train_Y, test_Y = split_build_dataset(
        df, test_ratio, target_col, seq_len)
    return model_dl_predict(train_X, train_Y, test_X, test_Y, model)


# =========================================================================== #
#  DL: KFold Analysis
# =========================================================================== #

def model_dl_analysis_with_KFold(df: pd.DataFrame, model_builder,
                                 test_ratio: float, target_col: str,
                                 n_folds: int, seq_len: int,
                                 shuffle: bool = False):
    """
    K-Fold 교차검증 DL 분석.

    Parameters
    ----------
    model_builder : 매 fold마다 새 모델을 반환하는 callable (가중치 초기화를 위해)

    Returns
    -------
    kf_scores : [MAE, MAPE, MSE, RMSE, MSLE, MBE, R2, AvgTime] 평균값
    kf_hists  : 각 지표별 fold 이력 리스트
    """
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    fold_scores, fold_times = [], []

    for _ in kf.split(df):
        model = model_builder()
        actual, preds, elapsed = model_dl_analysis_single(
            df, model, test_ratio, target_col, seq_len)
        fold_scores.append(model_sk_metrics(actual, preds))
        fold_times.append(elapsed)

    return _aggregate_kfold_scores(fold_scores, fold_times)
