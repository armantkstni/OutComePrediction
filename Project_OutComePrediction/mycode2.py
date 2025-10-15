import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ------------------ SETTINGS ------------------ #
SEED = 42
DATA_PATH = "labeled_logs_csv_processed/sepsis_cases_1.csv"
CASE_ID_COL = "case_id"
LABEL_COL = "label"
EVENT_COL = "event_nr"

encoding_methods = ["agg", "last", "index", "static", "previous", "bool"]
bucketing_methods = ["prefix", "state", "knn", "cluster", "base", "no"]

# ------------------ HELPER FUNCTIONS ------------------ #
def is_lstm_compatible(encoder_name):
    return encoder_name.lower() in ["index", "previous", "bool"]

def load_data(path):
    df = pd.read_csv(path,sep=';')
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in dataset")
    return df

def prepare_data(df):
    df = df.sort_values(by=[CASE_ID_COL, EVENT_COL])
    df["label"] = pd.factorize(df["label"])[0]
    return df

def encode_data(df, method):
    # Example logic — real implementation depends on your encoders
    if method == "agg":
        return df.select_dtypes(include=[np.number])
    elif method == "index":
        grouped = df.groupby(CASE_ID_COL)
        sequences = grouped.apply(lambda x: x.select_dtypes(include=[np.number]).values)
        return sequences.tolist()
    else:
        return df.select_dtypes(include=[np.number])

def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        padded[i, :length, :] = seq[:length]
    return padded

def build_lstm_model(hp, input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    for i in range(hp.Int("num_layers", 1, 2)):
        model.add(LSTM(units=hp.Int(f"units_{i}", 32, 128, step=32), return_sequences=(i == 0)))
        model.add(Dropout(rate=hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_and_evaluate(X_train, X_test, y_train, y_test, is_lstm=False, input_shape=None):
    if is_lstm:
        tuner = RandomSearch(
            lambda hp: build_lstm_model(hp, input_shape),
            objective='val_accuracy',
            max_trials=3,
            executions_per_trial=1,
            directory='lstm_tuning',
            project_name='lstm_opt')
        tuner.search(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)
        model = tuner.get_best_models(1)[0]
    else:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_test, y_pred_bin),
        'precision': precision_score(y_test, y_pred_bin),
        'recall': recall_score(y_test, y_pred_bin),
        'f1': f1_score(y_test, y_pred_bin),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }

def plot_results(results):
    df = pd.DataFrame(results)
    plt.figure(figsize=(16, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        plt.figure()
        sns.barplot(x='combination', y=metric, data=df)
        plt.title(f'Comparison of {metric.upper()} across all combinations')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"result_{metric}.png")

# ------------------ MAIN LOOP ------------------ #
data = load_data(DATA_PATH)
data = prepare_data(data)

results = []

for enc in encoding_methods:
    for buck in bucketing_methods:
        print(f"🔄 Running: encoding = {enc}, bucketing = {buck}")

        try:
            X = encode_data(data, enc)
            y = data[LABEL_COL].values

            if is_lstm_compatible(enc):
                maxlen = 10
                X = pad_sequences(X, maxlen=maxlen)
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)
                metrics = train_and_evaluate(X_train, X_test, y_train, y_test, is_lstm=True, input_shape=X.shape[1:])
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)
                metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

            metrics['combination'] = f"{enc}+{buck}"
            results.append(metrics)

        except Exception as e:
            print(f" Failed for {enc}+{buck}: {e}")

# ------------------ VISUALIZATION ------------------ #
plot_results(results)
print(" Done. Results plotted.")