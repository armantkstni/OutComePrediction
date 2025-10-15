import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras_tuner import RandomSearch
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.EncoderFactory import get_encoder
from experiments.BucketFactory import get_bucketer


# -------------------------- Config --------------------------
ENCODINGS = ["agg", "last", "index", "static", "bool", "previous"]
BUCKETINGS = ["single", "prefix", "state", "cluster", "knn", "base"]
DATA_PATH = "labeled_logs_csv_processed/sepsis_cases_1.csv"
CASE_ID_COL = "case id"
LABEL_COL = "label"
TIMESTAMP_COL = "time:timestamp"
DYNAMIC_CAT_COLS = ["activity"]
DYNAMIC_NUM_COLS = ["crp", "lacticacid", "leucocytes"]
MAX_EVENTS = 10

# ---------------------- Utility Functions ----------------------
def is_sequence_based(encoding):
    return encoding in ["index", "previous"]

def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int("units", 32, 128, step=32), input_shape=(MAX_EVENTS, len(DYNAMIC_NUM_COLS)), return_sequences=False))
    model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_mlp_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

def plot_bucket_distribution(buckets):
    plt.figure(figsize=(10, 5))
    sns.histplot(buckets, bins=20, kde=False)
    plt.title("Bucket Size Distribution")
    plt.xlabel("Bucket Size")
    plt.ylabel("Frequency")
    plt.savefig("outputs/bucket_size_distribution.png")
    plt.close()

def plot_case_length_distribution(data, label_col):
    plt.figure(figsize=(10, 5))
    sns.histplot(data[data[label_col] == 1]["event_nr"], color="red", label="Positive", kde=True)
    sns.histplot(data[data[label_col] == 0]["event_nr"], color="blue", label="Negative", kde=True)
    plt.legend()
    plt.title("Case Length Distribution")
    plt.xlabel("Event Count")
    plt.ylabel("Frequency")
    plt.savefig("outputs/case_length_distribution.png")
    plt.close()

# ---------------------- Main Loop ----------------------
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    raw_data = pd.read_csv(DATA_PATH)
    raw_data.columns = [c.lower() for c in raw_data.columns]

    results = []

    for encoding in ENCODINGS:
        for bucketing in BUCKETINGS:
            try:
                print(f"\U0001F501 Running: encoding = {encoding}, bucketing = {bucketing}")

                encoder = get_encoder(method=encoding,
                                       case_id_col=CASE_ID_COL,
                                       dynamic_cat_cols=DYNAMIC_CAT_COLS,
                                       dynamic_num_cols=DYNAMIC_NUM_COLS,
                                       max_events=MAX_EVENTS)

                bucketer = get_bucketer(method=bucketing,
                                         encoding_method=encoding,
                                         case_id_col=CASE_ID_COL,
                                         cat_cols=DYNAMIC_CAT_COLS,
                                         num_cols=DYNAMIC_NUM_COLS,
                                         n_clusters=5,
                                         random_state=42,
                                         n_neighbors=5)

                buckets = bucketer.fit_predict(raw_data)
                plot_bucket_distribution(pd.Series(buckets))
                plot_case_length_distribution(raw_data, label_col=LABEL_COL)

                features, labels = encoder.fit_transform(raw_data)

                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)

                if is_sequence_based(encoding):
                    tuner = RandomSearch(
                        build_lstm_model,
                        objective="val_accuracy",
                        max_trials=3,
                        executions_per_trial=1,
                        directory="tuner",
                        project_name=f"lstm_{encoding}_{bucketing}"
                    )
                    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])
                    best_model = tuner.get_best_models(1)[0]
                else:
                    best_model = build_mlp_model(X_train.shape[1])
                    best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)], verbose=0)

                y_prob = best_model.predict(X_test).ravel()
                y_pred = (y_prob > 0.5).astype(int)
                metrics = evaluate_model(y_test, y_pred, y_prob)
                metrics.update({"encoding": encoding, "bucketing": bucketing})
                results.append(metrics)

            except Exception as e:
                print(f" Failed for {encoding}+{bucketing}: {e}")
                continue

    df_results = pd.DataFrame(results)
    df_results.to_csv("outputs/model_results.csv", index=False)

    # Plot heatmap of AUC
    pivot_auc = df_results.pivot(index="encoding", columns="bucketing", values="auc")
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="coolwarm")
    plt.title("AUC Heatmap (All Methods)")
    plt.savefig("outputs/auc_heatmap.png")
    plt.close()
