import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "labeled_logs_csv_processed/sepsis_cases_1.csv"
CASE_ID_COL = "Case ID"
EVENT_COL = "event_nr"
LABEL_COL = "label"

encoding_methods = ["agg", "last", "index"]
bucketing_methods = ["prefix", "state", "knn"]

SEED = 42
np.random.seed(SEED)


df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()

def bucket_prefix(df):
    df["bucket_id"] = df[CASE_ID_COL] + "_len_" + df[EVENT_COL].astype(str)
    return df

def bucket_state(df):
    df["bucket_id"] = df["Activity"]
    return df

def bucket_knn(df, k=5):
    np.random.seed(42)
    df["bucket_id"] = np.random.randint(0, k, size=len(df))
    return df

bucketing_funcs = {
    "prefix": bucket_prefix,
    "state": bucket_state,
    "knn": bucket_knn
}

def encode_agg(df):
    agg_df = df.groupby(CASE_ID_COL).mean().reset_index()
    return agg_df

def encode_last(df):
    last_df = df.groupby(CASE_ID_COL).last().reset_index()
    return last_df

def encode_index(df):
    idx_df = df.copy()
    idx_df["event_index"] = idx_df.groupby(CASE_ID_COL).cumcount()
    return idx_df

encoding_funcs = {
    "agg": encode_agg,
    "last": encode_last,
    "index": encode_index
}


def build_model(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

bucket_results = []

for buck in bucketing_methods:
    for enc in encoding_methods:
        print(f"🔁 Running {buck} + {enc}")
        
        df_copy = df.copy()
        df_bucketed = bucketing_funcs[buck](df_copy)
        df_encoded = encoding_funcs[enc](df_bucketed)

        # حذف ستون غیرعددی برای مدل
        X = df_encoded.select_dtypes(include=[np.number])
        y = df_encoded[LABEL_COL].astype(int).values
        
        if X.shape[0] != len(y):
            continue
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )

        model = build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
        
        # ذخیره اندازه بکت‌ها
        bucket_sizes = df_bucketed.groupby("bucket_id")[CASE_ID_COL].nunique()
        bucket_results.append(pd.DataFrame({
            "log_bucket_size": np.log10(bucket_sizes),
            "combination": f"{buck}+{enc}"
        }))

# ==================== رسم نمودار Bucket Size Distributions ==================== #
bucket_df = pd.concat(bucket_results)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=bucket_df, x="log_bucket_size", hue="combination", fill=True, alpha=0.3)
plt.title("Bucket Size Distributions (log10 scale)")
plt.xlabel("log10(bucket size)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# ==================== Case Length Histogram ==================== #
case_lengths = df.groupby(CASE_ID_COL).size().reset_index(name="length")
case_labels = df.groupby(CASE_ID_COL)[LABEL_COL].first().reset_index()
case_info = pd.merge(case_lengths, case_labels, on=CASE_ID_COL)

plt.figure(figsize=(8, 5))
sns.histplot(case_info[case_info[LABEL_COL] == 1]["length"], color="red", label="Positive", kde=False)
sns.histplot(case_info[case_info[LABEL_COL] == 0]["length"], color="blue", label="Negative", kde=False)
plt.title("Case Length Histogram by Class")
plt.xlabel("Case Length (# events)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()
