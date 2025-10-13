import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
DATA_PATH = "labeled_logs_csv_processed/sepsis_cases_1.csv"
CASE_ID_COL = "case_id"
LABEL_COL = "label"
EVENT_COL = "event_nr"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

encoding_methods = ["agg", "last", "index"]
bucketing_methods = ["prefix", "state", "knn"] # simplified for deep models

# encoding_methods = ["agg", "last", "index", "static", "previous", "bool"]
# bucketing_methods = ["prefix", "state", "knn", "cluster", "base", "no"]
# ------------------ DATA FUNCTIONS ------------------ #
def load_data(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df[LABEL_COL] = pd.factorize(df[LABEL_COL])[0]
    return df.sort_values(by=[CASE_ID_COL, EVENT_COL])

def encode_data(df, method):
    if method == "agg" or method == "last":
        return df.select_dtypes(include=[np.number])
    elif method == "index":
        grouped = df.groupby(CASE_ID_COL)
        sequences = grouped.apply(lambda x: x.select_dtypes(include=[np.number]).values).tolist()
        return sequences
    else:
        return df.select_dtypes(include=[np.number])

def pad_sequences(sequences, maxlen):
    feature_dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), maxlen, feature_dim))
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length, :] = seq[:length]
    return padded

def build_lstm_model(hp, input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(hp.Int("units", 32, 128, step=32), return_sequences=False))
    model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(X_train, X_test, y_train, y_test, is_lstm=False, input_shape=None):
    if is_lstm:
        tuner = RandomSearch(
            lambda hp: build_lstm_model(hp, input_shape),
            objective='val_accuracy',
            max_trials=3, executions_per_trial=1,
            directory='lstm_tuning', project_name='sepsis')
        tuner.search(X_train, y_train, validation_split=0.2, epochs=5, verbose=0)
        model = tuner.get_best_models(1)[0]
    else:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32,
              validation_split=0.2, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_test, y_pred_bin),
        'precision': precision_score(y_test, y_pred_bin),
        'recall': recall_score(y_test, y_pred_bin),
        'f1': f1_score(y_test, y_pred_bin),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }


df = load_data(DATA_PATH)
results = []

for enc in encoding_methods:
    for buck in bucketing_methods:
        print(f"🔁 Running: encoding = {enc}, bucketing = {buck}")
        try:
            if enc == "index":
                X = encode_data(df, enc)
                y = df.drop_duplicates(subset=[CASE_ID_COL])[LABEL_COL].values
                X = pad_sequences(X, maxlen=10)
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)
                metrics = train_model(X_train, X_test, y_train, y_test, is_lstm=True, input_shape=X.shape[1:])
            else:
                X = encode_data(df, enc)
                y = df[LABEL_COL].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)
                metrics = train_model(X_train, X_test, y_train, y_test)
            metrics['combination'] = f"{enc}+{buck}"
            results.append(metrics)
        except Exception as e:
            print(f" Failed for {enc}+{buck}: {e}")


df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(RESULTS_DIR, "results_metrics.csv"), index=False)

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    plt.figure(figsize=(10, 5))
    sns.barplot(x='combination', y=metric, data=df_results)
    plt.title(f'{metric.upper()} across combinations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_barplot.png"))
    plt.close()


for buck in bucketing_methods:

    if buck == "prefix":
        buckets = df.groupby([CASE_ID_COL]).size()
    elif buck == "state":
        buckets = df.groupby([CASE_ID_COL]).size() 
    elif buck == "knn":
        buckets = df.groupby([CASE_ID_COL]).size()  
    else:
        continue
    
    plt.figure(figsize=(8, 5))
    sns.histplot(buckets, kde=True, stat="density", bins=20)
    plt.title(f"Bucket Size Distribution - {buck}")
    plt.xlabel("Number of Events in Bucket")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"bucket_size_distribution_{buck}.png"))
    plt.close()




case_lengths = df.groupby(CASE_ID_COL).size().reset_index(name="length")
case_labels = df.groupby(CASE_ID_COL)[LABEL_COL].first().reset_index(name="label")
case_info = pd.merge(case_lengths, case_labels, on=CASE_ID_COL)
print(case_info.groupby("label")["length"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(data=case_info, x="length", hue="label", bins=20, multiple="stack")
plt.title("Case Length Histogram (Positive vs Negative)")
plt.xlabel("Number of Events in Case")
plt.ylabel("Count")
plt.legend(title="Class", labels=["Negative", "Positive"])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "case_length_histogram.png"))
plt.close()

print("✅ All results and plots saved in 'results/' folder.")