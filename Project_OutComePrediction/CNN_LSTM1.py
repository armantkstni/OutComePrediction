# pipeline_full_optuna.py
"""
یکپارچه: Bucketing × Encoding × Model search با Optuna
- سازگار با کد اولیه‌ی کاربر و پیشنهادات بعدی
- خروجی‌ها در پوشه outputs/ ذخیره می‌شوند
"""

import os
import json
import math
import time
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score,
                             precision_recall_curve, precision_score, recall_score,
                             accuracy_score, matthews_corrcoef)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Masking, Conv1D, MaxPooling1D, BatchNormalization,
                                     LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

try:
    from river.drift import ADWIN
    RIVER_AVAILABLE = True
except Exception:
    RIVER_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# -------------------- CONFIG --------------------
DATA_PATH = "labeled_logs_csv_processed/BPIC17_O_Accepted.csv"  # مسیر پیش‌فرض؛ در صورت لزوم ویرایش کن
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CASE_ID_COL = "case_id"
EVENT_COL = "event_nr"
LABEL_COL = "label"

# ستون‌های کاتگوریک و عددی داینامیک (مثال — با داده‌ی شما باید تنظیم شود)
DYNAMIC_CAT_COLS = ["activity"]
DYNAMIC_NUM_COLS = ["crp", "lacticacid", "leucocytes"]  # اگر در دیتاست نیستند، کد آنها را کامل می‌کند

MAX_EVENTS = 20    # طول ماکسیمم پیشوند (sequence length)
SEED = 42

# -------------------- Utility --------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def add_time_features(df, timestamp_col=None):
    df = df.copy()
    # اگر time stamp داری میتونی timestamp_col را بدهی؛ در غیر این صورت از evnr استفاده کن
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([CASE_ID_COL, timestamp_col])
        df['elapsed'] = df.groupby(CASE_ID_COL)[timestamp_col].transform(lambda x: (x - x.iloc[0]).dt.total_seconds())
        df['evtime'] = df.groupby(CASE_ID_COL)[timestamp_col].diff().dt.total_seconds().fillna(0)
    else:
        if EVENT_COL not in df.columns:
            df[EVENT_COL] = df.groupby(CASE_ID_COL).cumcount() + 1
        df = df.sort_values([CASE_ID_COL, EVENT_COL])
        df['elapsed'] = df.groupby(CASE_ID_COL).cumcount()
        df['evtime'] = 0
    df['evnr'] = df.groupby(CASE_ID_COL).cumcount() + 1
    return df

# -------------------- Bucketers --------------------
class NoBucketer:
    def fit(self, X): return self
    def predict(self, X):
        # همه را یک bucket میدهد
        return np.zeros(len(X), dtype=int)

class PrefixLengthBucketer:
    def fit(self, X):
        # ثبت نکردن چیز خاصی نیاز نیست
        return self
    def predict(self, X):
        # bucket = طول پیشوند (evnr)
        if EVENT_COL in X.columns:
            return X[EVENT_COL].fillna(1).astype(int).to_numpy()
        else:
            return np.ones(len(X), dtype=int)

class KMeansBucketer:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.km = None
        self.encoder_feats = None

    def fit(self, X):
        # برای خوشه‌بندی از خلاصه‌برداری عددی استفاده می‌کنیم (agg)
        df_num = X.select_dtypes(include=[np.number]).fillna(0)
        self.encoder_feats = df_num.columns.tolist()
        if len(df_num) < self.n_clusters:
            self.km = KMeans(n_clusters=max(1, len(df_num)), random_state=self.random_state)
        else:
            self.km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.km.fit(df_num)
        return self

    def predict(self, X):
        df_num = X.select_dtypes(include=[np.number]).fillna(0)
        return self.km.predict(df_num)

class KNNBucketer:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.pipeline = None
    def fit(self, X):
        # استفاده از آخرین وضعیت برای encoding ساده
        df_enc = X.groupby(CASE_ID_COL).last().select_dtypes(include=[np.number]).fillna(0)
        self.X_enc = df_enc.values
        # labels as indices (dummy)
        self.labels = np.arange(len(df_enc))
        from sklearn.neighbors import KNeighborsClassifier
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(self.X_enc, self.labels)
        return self
    def predict(self, X):
        df_enc = X.groupby(CASE_ID_COL).last().select_dtypes(include=[np.number]).fillna(0)
        return self.knn.predict(df_enc)

# -------------------- Encoders --------------------
class AggregationEncoder:
    def fit_transform(self, df):
        # فقط ستون‌های عددی رو نگه دار
        df_num = df.select_dtypes(include=[np.number])
        agg = df_num.groupby(df[CASE_ID_COL]).agg(['mean', 'std', 'min', 'max']).fillna(0)
        # flatten MultiIndex
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        labels = df.groupby(CASE_ID_COL)[LABEL_COL].first().values
        return agg.reset_index(drop=True), labels


class LastStateEncoder:
    def fit_transform(self, df):
        last = df.groupby(CASE_ID_COL).last().reset_index(drop=True)
        # drop text cols
        labels = last[LABEL_COL].values
        X = last.drop(columns=[LABEL_COL], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        return X.values, labels

class StaticEncoder:
    # فرض بر این است که ستون‌های استاتیک قبل از هر ردیف event تکرار شده‌اند (مثل case attributes)
    def fit_transform(self, df):
        static = df.groupby(CASE_ID_COL).first().reset_index(drop=True)
        labels = static[LABEL_COL].values
        X = static.drop(columns=[LABEL_COL], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
        return X.values, labels

class IndexEncoder:
    # index encoding -> produce sequences (n_prefixes x MAX_EVENTS x feat_dim) and labels
    def __init__(self, cat_cols=None, num_cols=None, max_events=MAX_EVENTS):
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.max_events = max_events
        self.cat_encoders = {}

    def fit(self, df):
        # fit label encoders for categorical dynamic cols
        for c in self.cat_cols:
            le = LabelEncoder()
            df[c] = df[c].fillna("NA").astype(str)
            le.fit(df[c])
            self.cat_encoders[c] = le
        return self

    def transform(self, df):
        sequences = []
        labels = []
        case_ids = []
        grouped = df.groupby(CASE_ID_COL)
        for cid, group in grouped:
            group = group.sort_values(EVENT_COL)
            # for each prefix from 1..min(len, max_events)
            n_events = len(group)
            for prefix_len in range(1, min(n_events, self.max_events)+1):
                prefix = group.iloc[:prefix_len]
                per_event_feats = []
                for _, row in prefix.iterrows():
                    feats = []
                    # cat cols encoded as integers
                    for c in self.cat_cols:
                        val = row.get(c, "NA")
                        val = str(val) if not pd.isna(val) else "NA"
                        feats.append(float(self.cat_encoders[c].transform([val])[0]))
                    # numeric cols
                    for c in self.num_cols:
                        feats.append(float(row.get(c, 0.0) if not pd.isna(row.get(c, 0.0)) else 0.0))
                    # synthetic time features
                    feats.append(float(row.get('elapsed', 0.0)))
                    feats.append(float(row.get('evtime', 0.0)))
                    feats.append(float(row.get('evnr', 0.0)))
                    per_event_feats.append(feats)
                # pad
                feat_dim = len(per_event_feats[0])
                seq = np.zeros((self.max_events, feat_dim), dtype=float)
                seq[:len(per_event_feats), :] = np.array(per_event_feats)
                sequences.append(seq)
                labels.append(int(prefix[LABEL_COL].iloc[0]))
                case_ids.append(cid)
        return np.array(sequences), np.array(labels), np.array(case_ids)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

# -------------------- Models --------------------
def build_cnn_lstm(input_shape, conv_filters=64, kernel_size=3, lstm_units=128, dropout=0.4, lr=1e-3, use_focal=False):
    inp = Input(shape=input_shape)
    x = Masking(mask_value=0.0)(inp)
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    loss = "binary_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=loss, metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def build_bilstm(input_shape, lstm_units=128, dropout=0.4, lr=1e-3):
    inp = Input(shape=input_shape)
    x = Masking(mask_value=0.0)(inp)
    x = Bidirectional(LSTM(lstm_units))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def build_mlp(input_shape, hidden=64, dropout=0.3, lr=1e-3):
    inp = Input(shape=(input_shape,))
    x = Dense(hidden, activation='relu')(inp)
    x = Dropout(dropout)(x)
    x = Dense(hidden//2, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# -------------------- Drift detector (ADWIN or Page-Hinkley) --------------------
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
            print("river not available — using PageHinkley fallback.")

    def update_and_check(self, val):
        if self.use_river:
            self.detector.update(val)
            return self.detector.change_detected
        else:
            return self.detector.update(val)

# -------------------- Helpers: threshold tuning & evaluation --------------------
def choose_best_threshold(y_val, y_prob):
    # choose threshold maximizing F1 on validation
    prec, rec, ths = precision_recall_curve(y_val, y_prob)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    idx = np.nanargmax(f1s)
    best_th = ths[idx] if idx < len(ths) else 0.5
    best_f1 = f1s[idx]
    return float(best_th), float(best_f1)

def eval_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true))>1 else float("nan")
    }
    return metrics

# -------------------- Main pipeline for one combo --------------------
def prepare_features_for_combo(df, bucketer_name, encoder_name, index_encoder_obj=None, bucket_as_feature=True):
    """
    Returns features X, labels y, and optionally case_ids for sequence encoders.
    For index encoder -> returns sequences (n_samples, MAX_EVENTS, feat_dim), labels, case_ids
    For non-index encoders -> returns X (n_samples, feat_dim), labels
    """

    # 1) compute bucket assignments for each (case,prefix)
    if bucketer_name == "none":
        buckets = np.zeros(len(df), dtype=int)
    elif bucketer_name == "prefix":
        buckets = df[EVENT_COL].fillna(1).astype(int).to_numpy()
    elif bucketer_name == "kmeans":
        km = KMeans(n_clusters=5, random_state=SEED)
        num = df.select_dtypes(include=[np.number]).fillna(0)
        if len(num) < 5:
            buckets = np.zeros(len(df), dtype=int)
        else:
            try:
                km.fit(num)
                buckets = km.predict(num)
            except Exception:
                buckets = np.zeros(len(df), dtype=int)
    elif bucketer_name == "state":
    # bucket by last activity of case
        last_state = df.groupby(CASE_ID_COL).tail(1)[[CASE_ID_COL] + DYNAMIC_CAT_COLS]
        state_col = DYNAMIC_CAT_COLS[0] if DYNAMIC_CAT_COLS else CASE_ID_COL
        map_state = dict(zip(last_state[CASE_ID_COL], last_state[state_col].astype(str)))
        # مقادیر متنی رو با LabelEncoder به اعداد تبدیل کن
        states = df[CASE_ID_COL].map(map_state).fillna("NA").astype(str)
        le = LabelEncoder()
        buckets = le.fit_transform(states)
    else:
        buckets = np.zeros(len(df), dtype=int)

    # add bucket as a column if needed
    df2 = df.copy()
    df2["_bucket"] = buckets

    # Encoding
    if encoder_name == "index":
        # Index encoder: sequences per prefix
        idx_enc = index_encoder_obj or IndexEncoder(cat_cols=DYNAMIC_CAT_COLS, num_cols=DYNAMIC_NUM_COLS, max_events=MAX_EVENTS)
        seqs, labels, case_ids = idx_enc.fit_transform(df2)
        # if bucket_as_feature -> add bucket as extra numeric channel (replicate scalar across timesteps)
        if bucket_as_feature:
            # we need bucket for each prefix -> compute mapping from (case_id,prefix_len) to bucket scalar
            # Construct an array of bucket scalar for each produced prefix in same order as seqs
            # The IndexEncoder returns prefixes in grouped case order; we can recompute the same order:
            bucket_list = []
            grouped = df2.groupby(CASE_ID_COL)
            for cid, group in grouped:
                group = group.sort_values(EVENT_COL)
                n_events = len(group)
                for prefix_len in range(1, min(n_events, MAX_EVENTS)+1):
                    # bucket value at prefix end = group.iloc[prefix_len-1]["_bucket"]
                    bucket_list.append(float(group.iloc[prefix_len-1]["_bucket"]))
            bucket_array = np.array(bucket_list)
            # normalize bucket values
            if bucket_array.max() - bucket_array.min() > 0:
                bucket_array = (bucket_array - bucket_array.min()) / (bucket_array.max() - bucket_array.min())
            # append as extra feature channel (replicate along time axis)
            bucket_channel = bucket_array.reshape(-1, 1, 1) * np.ones((1, MAX_EVENTS, 1))
            seqs = np.concatenate([seqs, bucket_channel], axis=2)
        return seqs, labels, case_ids
    else:
        # Non-sequence encoders -> produce one row per case (we will use last/agg/static)
        if encoder_name == "agg":
            X_df, labels = AggregationEncoder().fit_transform(df2)
            X = X_df.values
        elif encoder_name == "last":
            X, labels = LastStateEncoder().fit_transform(df2)
        elif encoder_name == "static":
            X, labels = StaticEncoder().fit_transform(df2)
        else:
            X, labels = LastStateEncoder().fit_transform(df2)

        # add bucket as categorical numeric column if desired
        if bucket_as_feature:
            # map per-case bucket: take bucket value of last event of that case
            bucket_per_case = df2.groupby(CASE_ID_COL)["_bucket"].last().reset_index(drop=True).astype(float)
            # normalize
            if bucket_per_case.max() - bucket_per_case.min() > 0:
                bucket_per_case = (bucket_per_case - bucket_per_case.min()) / (bucket_per_case.max() - bucket_per_case.min())
            X = np.hstack([X, bucket_per_case.values.reshape(-1,1)])
        return X, labels

# -------------------- Single trial training & eval --------------------
def train_and_eval_model(X_train, y_train, X_val, y_val, model_type, params, save_prefix):
    """
    model_type: 'cnn_lstm' (for sequences), 'bilstm' (for sequences), 'mlp' (for tabular)
    params: dict contains hyperparams: lstm_units, conv_filters, kernel_size, dropout, lr, batch_size, epochs, class_weight_pos
    """
    # Determine input shape
    if X_train.ndim == 3:
        input_shape = X_train.shape[1:]
    else:
        input_shape = (X_train.shape[1],)

    # build model
    if model_type == "cnn_lstm":
        model = build_cnn_lstm(input_shape, conv_filters=params.get('conv_filters',64),
                               kernel_size=params.get('kernel_size',3),
                               lstm_units=params.get('lstm_units',128),
                               dropout=params.get('dropout',0.4),
                               lr=params.get('lr',1e-3))
    elif model_type == "bilstm":
        model = build_bilstm(input_shape, lstm_units=params.get('lstm_units',128),
                             dropout=params.get('dropout',0.4),
                             lr=params.get('lr',1e-3))
    else:
        model = build_mlp(input_shape[0], hidden=params.get('hidden',64),
                          dropout=params.get('dropout',0.3),
                          lr=params.get('lr',1e-3))

    # callbacks & checkpoint (.keras suffix)
    model_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=0),
        ModelCheckpoint(model_file, monitor='val_auc', mode='max', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, verbose=0)
    ]

    # class weights
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {i: float(w) for i,w in enumerate(cw)}
    # if user provided factor for positive class
    if params.get('class_weight_pos', None):
        cw_dict[1] = float(params['class_weight_pos'])
        cw_dict[0] = 1.0

    # train
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=int(params.get('epochs', 20)),
                        batch_size=int(params.get('batch_size', 64)),
                        class_weight=cw_dict,
                        callbacks=callbacks,
                        verbose=0)

    # choose best threshold on val
    y_val_prob = model.predict(X_val, batch_size=int(params.get('batch_size',64))).flatten()
    best_th, best_f1 = choose_best_threshold(y_val, y_val_prob)

    # evaluate on val & return model + threshold
    val_metrics = eval_metrics(y_val, y_val_prob, threshold=best_th)
    val_metrics['best_threshold'] = best_th
    val_metrics['best_f1_on_val'] = best_f1
    return model, val_metrics

# -------------------- Optuna objective --------------------
def objective_optuna(trial, df):
    # Suggest bucketer / encoder / model types
    bucketer = trial.suggest_categorical("bucketer", ["none", "prefix", "kmeans", "state"])
    encoder = trial.suggest_categorical("encoder", ["index", "last", "agg", "static"])
    model_type = trial.suggest_categorical("model_type", ["bilstm", "cnn_lstm"])

    # hyperparams
    params = {
        "lstm_units": trial.suggest_categorical("lstm_units", [64, 128, 256]),
        "conv_filters": trial.suggest_categorical("conv_filters", [32, 64]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2,3,5]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "epochs": trial.suggest_int("epochs", 5, 30),
        "class_weight_pos": trial.suggest_float("class_weight_pos", 1.0, 10.0),
        "threshold": trial.suggest_float("threshold", 0.3, 0.7)
    }

    # prepare features for combo
    index_enc_obj = None
    if encoder == "index":
        index_enc_obj = IndexEncoder(cat_cols=DYNAMIC_CAT_COLS, num_cols=DYNAMIC_NUM_COLS, max_events=MAX_EVENTS)
        X, y, case_ids = prepare_features_for_combo(df, bucketer, encoder, index_encoder_obj=index_enc_obj, bucket_as_feature=True)
    else:
        X, y = prepare_features_for_combo(df, bucketer, encoder, index_encoder_obj=None, bucket_as_feature=True)

    # compatibility check
    if encoder == "index" and model_type == "mlp":
        raise optuna.TrialPruned()

    if encoder in ["static", "last", "agg"] and model_type in ["cnn_lstm", "bilstm"]:
        raise optuna.TrialPruned()


    # if index -> X is sequences (3D), else 2D
    # split by case to avoid leakage: if sequences produced, use case_ids; else split rows
    if encoder == "index":
        # stratified split by y but also by case -> use simple split: split unique cases
        unique_cases = np.unique(case_ids)
        train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=SEED)
        train_mask = np.isin(case_ids, train_cases)
        test_mask = np.isin(case_ids, test_cases)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        # further split X_train into train/val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

    # optionally apply SMOTE for tabular data
    if X_train.ndim == 2 and IMBLEARN_AVAILABLE:
        # balance only if very imbalanced
        pos_ratio = np.mean(y_train)
        if pos_ratio < 0.3:
            try:
                sm = SMOTE(random_state=SEED)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass

    # train model and evaluate
    save_prefix = f"trial_{trial.number}_{bucketer}_{encoder}_{model_type}"
    model, val_metrics = train_and_eval_model(X_train, y_train, X_val, y_val, model_type, params, save_prefix)

    # evaluate on test using threshold chosen from val
    best_th = val_metrics.get('best_threshold', params['threshold'])
    y_test_prob = model.predict(X_test, batch_size=int(params['batch_size'])).flatten()
    test_metrics = eval_metrics(y_test, y_test_prob, threshold=best_th)

    # objective score: weighted combination
    score = 0.7 * test_metrics.get('pr_auc', 0.0) + 0.3 * test_metrics.get('f1', 0.0)

    # optional: save trial results
    trial_result = {
        "trial": trial.number,
        "bucketer": bucketer,
        "encoder": encoder,
        "model_type": model_type,
        "params": params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "score": score
    }
    with open(os.path.join(OUTPUT_DIR, f"trial_{trial.number}_result.json"), "w") as f:
        json.dump(trial_result, f, indent=2)
        
    csv_file = os.path.join(OUTPUT_DIR, "all_results.csv")
    row = {
        "trial": trial.number,
        "bucketer": bucketer,
        "encoder": encoder,
        "model_type": model_type,
        **params,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "score": score
    }
    df_row = pd.DataFrame([row])
    if not os.path.exists(csv_file):
        df_row.to_csv(csv_file, index=False)
    else:
        df_row.to_csv(csv_file, mode="a", header=False, index=False)

    return score


# -------------------- Run search --------------------
def run_search(n_trials=30):
    # load data once
    df = safe_read_csv(DATA_PATH)
    df = add_time_features(df, timestamp_col=None)
    if LABEL_COL not in df.columns:
        raise ValueError(f"{LABEL_COL} column not found in data")
    # factorize label to 0/1
    df[LABEL_COL] = pd.factorize(df[LABEL_COL])[0]

    # ensure dynamic cols exist
    for c in DYNAMIC_CAT_COLS:
        if c not in df.columns:
            df[c] = "NA"
    for c in DYNAMIC_NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0

    if OPTUNA_AVAILABLE:
        study = optuna.create_study(direction="maximize")
        func = lambda trial: objective_optuna(trial, df)
        study.optimize(func, n_trials=n_trials)
        print("Best trial params:", study.best_params)
        print("Best trial value:", study.best_value)
        with open(os.path.join(OUTPUT_DIR, "optuna_best.json"), "w") as f:
            json.dump({"best_params": study.best_params, "best_value": float(study.best_value)}, f, indent=2)
    else:
        # fallback: grid search coarse
        print("Optuna not available — running fallback grid (coarse)")
        bucketers = ["none", "prefix", "kmeans", "state"]
        encoders = ["index", "last", "agg", "static"]
        model_types = ["bilstm", "cnn_lstm", "mlp"]
        best = {"score": -1}
        for b in bucketers:
            for e in encoders:
                for m in model_types:
                    # simple fixed params
                    params = {"lstm_units":128, "dropout":0.3, "lr":1e-3, "batch_size":64, "epochs":10, "class_weight_pos":3.0}
                    try:
                        # به جای شبیه‌سازی کامل Optuna، مستقیم objective رو صدا می‌زنیم
                        trial = SimpleNamespace(
                            number=0,
                            suggest_categorical=lambda name, choices: choices[0],
                            suggest_float=lambda name, low, high, **kwargs: (low+high)/2,
                            suggest_int=lambda name, low, high: (low+high)//2,
                            suggest_loguniform=lambda name, low, high: math.sqrt(low*high),
                        )
                        score = objective_optuna(trial, df)
                    except Exception:
                        score = -1
                    if score > best["score"]:
                        best = {"bucketer": b, "encoder": e, "model": m, "score": score}
        print("Fallback best:", best)

    print("Search finished. Results saved in", OUTPUT_DIR)

# -------------------- After search: drift monitoring example --------------------
def drift_monitor_and_retrain(best_model_path, df):
    # load model
    try:
        model = tf.keras.models.load_model(best_model_path)
    except Exception as e:
        print("Cannot load best model:", e)
        return
    dm = DriftManager()
    # create case-level last-prefix stream for test and simulate streaming
    df_sorted = df.sort_values([CASE_ID_COL, EVENT_COL])
    case_groups = list(df_sorted.groupby(CASE_ID_COL))
    buffer_X = []
    buffer_y = []
    for cid, group in case_groups:
        # take last prefix representation (use index encoding transform here)
        idx_enc = IndexEncoder(cat_cols=DYNAMIC_CAT_COLS, num_cols=DYNAMIC_NUM_COLS, max_events=MAX_EVENTS)
        seqs, labels, case_ids = idx_enc.fit_transform(group)
        if len(seqs)==0:
            continue
        last_seq = seqs[-1].reshape(1, seqs.shape[1], seqs.shape[2])
        prob = model.predict(last_seq)[0,0]
        change = dm.update_and_check(prob)
        if change:
            print(f"Drift detected at case {cid}. Retraining on buffered samples of size {len(buffer_X)}")
            if len(buffer_X) >= 10:
                X_buf = np.vstack(buffer_X)
                y_buf = np.hstack(buffer_y)
                # fine-tune for a few epochs
                try:
                    model.fit(X_buf, y_buf, epochs=3, batch_size=32, verbose=0)
                    print("Retrained model on buffered data.")
                except Exception as e:
                    print("Retrain failed:", e)
            buffer_X, buffer_y = [], []
        else:
            buffer_X.append(last_seq[0])
            buffer_y.append(labels[-1])
    print("Drift monitoring finished.")


def plot_results():
    csv_file = os.path.join(OUTPUT_DIR, "all_results.csv")
    if not os.path.exists(csv_file):
        print("No results CSV found.")
        return

    df = pd.read_csv(csv_file)
    combo = df["bucketer"] + "_" + df["encoder"] + "_" + df["model_type"]
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1",
               "test_auc", "test_pr_auc", "test_mcc"]

    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for m in metrics:
        if m not in df.columns:
            continue
        plt.figure(figsize=(12,6))
        plt.bar(combo, df[m])
        plt.xticks(rotation=90)
        plt.title(f"Comparison of {m} across combos")
        plt.ylabel(m)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{m}.png"))
        plt.close()
    print(f"Plots saved in {plot_dir}")

# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting search... (Optuna available:", OPTUNA_AVAILABLE, ", imblearn:", IMBLEARN_AVAILABLE, ", river:", RIVER_AVAILABLE, ")")
    run_search(n_trials=50)
    print("Done. Check outputs/ for trial results and logs.")
    plot_results()

