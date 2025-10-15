import os
import sys
import math
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)
from sklearn.utils import compute_class_weight

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout, Conv1D,
                                     MaxPooling1D, LSTM, Bidirectional,
                                     GlobalAveragePooling1D, BatchNormalization,
                                     Masking, TimeDistributed, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# try optional libs
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    from river.drift import ADWIN
    RIVER_AVAILABLE = True
except Exception:
    RIVER_AVAILABLE = False

# -------------------- CONFIG --------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "labeled_logs_csv_processed/sepsis_cases_1.csv"  
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CASE_ID_COL = "case_id"   
EVENT_COL = "event_nr"
LABEL_COL = "label"

MAX_EVENTS = 20  
BATCH_SIZE = 64
EPOCHS = 30


DYNAMIC_CAT_COLS = ["activity"]
DYNAMIC_NUM_COLS = ["crp", "lacticacid", "leucocytes"]  

# -------------------- Utility functions --------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=';')
    # normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def add_time_features(df, timestamp_col=None):

    df = df.copy()
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([CASE_ID_COL, timestamp_col])
        df['elapsed'] = df.groupby(CASE_ID_COL)[timestamp_col].transform(lambda x: (x - x.iloc[0]).dt.total_seconds())
        df['evtime'] = df.groupby(CASE_ID_COL)[timestamp_col].diff().dt.total_seconds().fillna(0)
        df['since_midnight'] = df[timestamp_col].dt.hour * 3600 + df[timestamp_col].dt.minute*60 + df[timestamp_col].dt.second
        df['hour'] = df[timestamp_col].dt.hour.astype(int)
        df['day'] = df[timestamp_col].dt.day.astype(int)
        df['month'] = df[timestamp_col].dt.month.astype(int)
    else:
        if EVENT_COL not in df.columns:
            df[EVENT_COL] = df.groupby(CASE_ID_COL).cumcount() + 1
        df['elapsed'] = df.groupby(CASE_ID_COL).cumcount()
        df['evtime'] = 0
        df['hour'] = 0
        df['day'] = 0
        df['month'] = 0
    df['evnr'] = df.groupby(CASE_ID_COL).cumcount() + 1
    return df

def make_index_sequences(df, maxlen=MAX_EVENTS, cat_cols=None, num_cols=None):
    """
    index encoding -> برای هر case، تمام پیشوندها را تولید کرده و برای هر پیشوند یک sequence (maxlen x features) برمی‌گردانیم.
    categorical columns should be label-encoded beforehand (integers).
    """
    sequences = []
    labels = []
    case_ids = []
    grouped = df.groupby(CASE_ID_COL)
    for case_id, group in grouped:
        n_events = len(group)
        for prefix_len in range(1, min(n_events, maxlen)+1):
            prefix = group.iloc[:prefix_len]
            row_feats = []
            for _, row in prefix.iterrows():
                feat = []
                # categorical as integer(s)
                if cat_cols:
                    for c in cat_cols:
                        feat.append(int(row[c]) if not pd.isna(row[c]) else 0)
                # numeric
                if num_cols:
                    for c in num_cols:
                        feat.append(float(row[c]) if not pd.isna(row[c]) else 0.0)
                # add synthetic numeric features (elapsed, evtime, evnr)
                feat.append(float(row.get('elapsed', 0.0)))
                feat.append(float(row.get('evtime', 0.0)))
                feat.append(float(row.get('evnr', 0)))
                row_feats.append(feat)
            # pad to maxlen
            seq = np.zeros((maxlen, len(row_feats[0])), dtype=float)
            seq[:len(row_feats), :] = np.array(row_feats)
            sequences.append(seq)
            labels.append(int(group[LABEL_COL].iloc[0]))
            case_ids.append(case_id)
    return np.array(sequences), np.array(labels), np.array(case_ids)

# focal loss implementation
import tensorflow.keras.backend as K
def focal_loss(gamma=2., alpha=.75):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal = weight * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(focal)
    return focal_loss_fixed

# simple Page-Hinkley drift detector (fallback)
class PageHinkley:
    def __init__(self, delta=0.005, lambda_thr=50, alpha=1-0.0001):
        self.delta = delta
        self.lambda_thr = lambda_thr
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.cum_sum = 0.0
        self.n = 0

    def update(self, x):
        self.n += 1
        self.mean = self.mean + (x - self.mean) / self.n
        self.cum_sum = self.alpha * self.cum_sum + (x - self.mean - self.delta)
        if self.cum_sum > self.lambda_thr:
            self.reset()
            return True
        return False

class DriftManager:
    def __init__(self):
        if RIVER_AVAILABLE:
            self.detector = ADWIN()
            self.use_river = True
            print("Using river.ADWIN for drift detection.")
        else:
            self.detector = PageHinkley()
            self.use_river = False
            print("river not available — using PageHinkley fallback for drift detection.")

    def update_and_check(self, value):
        if self.use_river:
            # ADWIN expects a float
            self.detector.update(value)
            return self.detector.change_detected
        else:
            return self.detector.update(value)

# -------------------- Model builders --------------------
def build_cnn_lstm_model(input_shape, conv_filters=64, kernel_size=3, lstm_units=128, dropout_rate=0.4, bidirectional=False):
    inp = Input(shape=input_shape, name="sequence_input")
    x = Masking(mask_value=0.0)(inp)
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    if bidirectional:
        x = Bidirectional(LSTM(lstm_units))(x)
    else:
        x = LSTM(lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model

def build_bilstm_model(input_shape, lstm_units=128, dropout_rate=0.3):
    inp = Input(shape=input_shape, name="sequence_input")
    x = Masking(mask_value=0.0)(inp)
    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model

# -------------------- Evaluation --------------------
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics['auc'] = float('nan')
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics['pr_auc'] = float('nan')
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics['mcc'] = float('nan')
    return metrics

# -------------------- Main pipeline --------------------
def main_pipeline(data_path=DATA_PATH):
    df = safe_read_csv(data_path)
    # make sure column names normalized
    df = add_time_features(df, timestamp_col=None)  
    # encode label
    if LABEL_COL not in df.columns:
        raise ValueError(f"{LABEL_COL} column not found in data")
    df[LABEL_COL] = pd.factorize(df[LABEL_COL])[0]
    print(df[LABEL_COL].value_counts(normalize=True))

    # Encode categorical dynamic features with LabelEncoder (for embedding we need integer ids)
    cat_encoders = {}
    for c in DYNAMIC_CAT_COLS:
        le = LabelEncoder()
        df[c] = df[c].fillna("NA")
        df[c] = le.fit_transform(df[c].astype(str))
        cat_encoders[c] = le
    # Fill numeric
    for c in DYNAMIC_NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # Create index sequences (sequence-based encoding)
    print("Creating index sequences (may take time)...")
    X_seq, y_all, case_ids = make_index_sequences(df, maxlen=MAX_EVENTS, cat_cols=DYNAMIC_CAT_COLS, num_cols=DYNAMIC_NUM_COLS)
    print(f"Sequences shape: {X_seq.shape}, labels: {y_all.shape}")

    # split by case to avoid leakage: unique case ids
    unique_cases = np.unique(case_ids)
    train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=SEED, stratify=None)

    train_mask = np.isin(case_ids, train_cases)
    test_mask = np.isin(case_ids, test_cases)
    X_train, X_test = X_seq[train_mask], X_seq[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]

    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    # standardize numeric features across last axes (but careful: features include encoded cat ints as first dims)
    # We'll scale entire feature vector per time-step except categorical positions (we'll assume cat cols are first positions)
    n_cat = len(DYNAMIC_CAT_COLS)
    n_num = len(DYNAMIC_NUM_COLS) + 3  # num cols + elapsed + evtime + evnr
    feat_dim = X_train.shape[2]
    assert feat_dim == n_cat + n_num

    # Scale numeric slices
    scaler = StandardScaler()
    # reshape numeric parts to 2D for scaler fit: (samples*time, n_num)
    train_num = X_train[:, :, n_cat:].reshape(-1, n_num)
    scaler.fit(train_num)
    # transform both train and test numeric parts
    X_train_num = scaler.transform(X_train[:, :, n_cat:].reshape(-1, n_num)).reshape(X_train.shape[0], X_train.shape[1], n_num)
    X_test_num = scaler.transform(X_test[:, :, n_cat:].reshape(-1, n_num)).reshape(X_test.shape[0], X_test.shape[1], n_num)
    # replace numeric slice
    X_train[:, :, n_cat:] = X_train_num
    X_test[:, :, n_cat:] = X_test_num

    # --- class weights for imbalance (for sample weighting) ---
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i,w in enumerate(class_weights)}
    print("Class weights:", cw_dict)

    # ---------------- Model training with drift detection & possible retrain ----------------
    # choose model type: 'cnn_lstm' or 'bilstm'
    model_type = "cnn_lstm"  
    bidir = False 

    if model_type == "cnn_lstm":
        model = build_cnn_lstm_model(input_shape=(MAX_EVENTS, feat_dim), conv_filters=64, kernel_size=3, lstm_units=128, dropout_rate=0.4, bidirectional=bidir)
    else:
        model = build_bilstm_model(input_shape=(MAX_EVENTS, feat_dim), lstm_units=128, dropout_rate=0.4)

    model.summary()

    # callbacks
    model_file = os.path.join(OUTPUT_DIR, f"best_model_{model_type}.keras")
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_file, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, class_weight={0:1, 1:5}, verbose=2)

    # evaluate initial model
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
    metrics = evaluate_metrics(y_test, y_prob)
    print("Initial evaluation:", metrics)

    # save metrics
    with open(os.path.join(OUTPUT_DIR, f"metrics_initial_{model_type}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------ concept drift monitoring on test stream ------------------
    dm = DriftManager()
    # we will stream test cases sequentially (by case) and update detector on predicted probability mean per case
    test_df_cases = defaultdict(list)
    for seq, y_true, cid in zip(X_test, y_test, case_ids[test_mask]):
        test_df_cases[cid].append((seq, y_true))
    # sort by case id to have deterministic order
    ordered_case_ids = list(test_df_cases.keys())

    drift_points = []
    buffer_X = []
    buffer_y = []
    buffer_size_for_retrain = 200  # after drift, collect this many samples to retrain
    for idx, cid in enumerate(ordered_case_ids):
        # for simplicity take the last prefix of the case as current observation
        seqs = test_df_cases[cid]
        seq_arr = np.array([s for s,_ in seqs])
        y_arr = np.array([yy for _,yy in seqs])
        # predict on the last prefix
        last_seq = seq_arr[-1].reshape(1, MAX_EVENTS, feat_dim)
        prob = model.predict(last_seq)[0,0]
        # feed detector with prob (or you could use feature stats)
        changed = dm.update_and_check(prob)
        if changed:
            print(f"Drift detected at case {cid} (index {idx}) — retraining will be triggered.")
            drift_points.append((idx, cid, prob))
            # gather buffer samples (most recent ones up to buffer size)
            # We'll collect previous N cases for retraining (if available)
            recent_case_ids = ordered_case_ids[max(0, idx-buffer_size_for_retrain):idx+1]
            X_buf = []
            y_buf = []
            for rc in recent_case_ids:
                # take last prefix for each case (simpler)
                sarr = np.array([s for s,_ in test_df_cases[rc]])
                yarr = np.array([yy for _,yy in test_df_cases[rc]])
                X_buf.append(sarr[-1])
                y_buf.append(yarr[-1])
            X_buf = np.array(X_buf)
            y_buf = np.array(y_buf)
            # combine with some training data (to avoid catastrophic forgetting)
            X_retrain = np.concatenate([X_train[:len(X_buf)], X_buf], axis=0)
            y_retrain = np.concatenate([y_train[:len(y_buf)], y_buf], axis=0)
            print(f"Retraining on {len(X_retrain)} samples (including historical).")
            # reinitialize / fine-tune model (we'll continue training existing model with lower lr)
            tf.keras.backend.set_value(model.optimizer.lr, 1e-4)
            model.fit(X_retrain, y_retrain, validation_split=0.1, epochs=5, batch_size=BATCH_SIZE, class_weight={0:1, 1:5}, verbose=2)
            # after retrain reset detector state if using PageHinkley or ADWIN internal reset handled
            # continue streaming
    print("Drift points:", drift_points)

    # final evaluation after possible retrains
    y_prob_final = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
    metrics_final = evaluate_metrics(y_test, y_prob_final)
    print("Final evaluation:", metrics_final)
    with open(os.path.join(OUTPUT_DIR, f"metrics_final_{model_type}.json"), "w") as f:
        json.dump(metrics_final, f, indent=2)

    # plot ROC/PR etc.
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob_final)
        prec, rec, _ = precision_recall_curve(y_test, y_prob_final)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC={metrics_final['auc']:.3f}")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{model_type}.png"))
        plt.close()

        plt.figure()
        plt.plot(rec, prec, label=f"PR AUC={metrics_final['pr_auc']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f"pr_{model_type}.png"))
        plt.close()
    except Exception as e:
        print("Could not plot ROC/PR:", e)

    print("All done. Outputs saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main_pipeline()


import itertools
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import optuna



BUCKETERS = ["NoBucketer", "PrefixLengthBucketer", "KMeansBucketer", "StateBucketer"]
ENCODERS = ["Static", "LastState", "Aggregation", "Index"]

def run_pipeline(bucket_method, encoder_method, params):

    bucketer = get_bucketer(bucket_method)
    encoder = get_encoder(encoder_method)
    X_train, y_train, X_val, y_val = prepare_data(bucketer, encoder)

    model = build_model(params, input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=params["epochs"], batch_size=params["batch_size"],
              verbose=0)

    y_prob = model.predict(X_val).flatten()
    y_pred = (y_prob >= params["threshold"]).astype(int)

    pr_auc = average_precision_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    return 0.7 * pr_auc + 0.3 * f1

def objective(trial):
    # انتخاب bucketing و encoding
    bucket_method = trial.suggest_categorical("bucket", BUCKETERS)
    encoder_method = trial.suggest_categorical("encoder", ENCODERS)


    params = {
        "lstm_units": trial.suggest_categorical("lstm_units", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "learning_rate": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 10, 30),
        "threshold": trial.suggest_float("threshold", 0.3, 0.7)
    }

    score = run_pipeline(bucket_method, encoder_method, params)
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best setting:", study.best_params)
print("Best score:", study.best_value)
