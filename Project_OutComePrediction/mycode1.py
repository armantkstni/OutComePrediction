import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

log_path = r"C:\Users\Arman Takestani\Downloads\Compressed\predictive-monitoring-benchmark-master\predictive-monitoring-benchmark-master\labeled_logs_csv_processed\sepsis_cases_1.csv"
data = pd.read_csv(log_path, sep=';')

prefix_length = 10
bucket_methods = ["single", "prefix", "state", "cluster", "knn"]
encodings = ["index", "agg", "laststate"]

data.columns = data.columns.str.strip().str.lower()

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])


data = data.sort_values(['case id', 'event_nr'])


activity_encoder = LabelEncoder()
activity_encoder.fit(data['activity'])


results = []


def create_index_sequences(group, prefix_length):
    sequences, labels = [], []
    events = group['activity'].tolist()
    label = group['label'].iloc[0]
    for i in range(1, min(len(events), prefix_length)+1):
        seq = events[:i]
        seq_encoded = [activity_encoder.transform([a])[0] for a in seq]
        sequences.append(seq_encoded)
        labels.append(label)
    return sequences, labels

for bucket in bucket_methods:
    for encoding in encodings:
        print(f"🔄 Running: bucket = {bucket}, encoding = {encoding}")
        X, y = [], []

        for _, group in data.groupby('case id'):
            if encoding == "index":
                seqs, labels = create_index_sequences(group, prefix_length)
                X.extend(seqs)
                y.extend(labels)
            elif encoding in ["agg", "laststate"]:
                row = group.iloc[-1]

                numeric_features = []
                for col, val in row.items():
                    if col == 'label':
                        continue
                    try:
                        numeric_features.append(float(val))
                    except:
                        continue

                X.append(numeric_features)
                y.append(row['label'])

        if encoding == "index":
            X = pad_sequences(X, maxlen=prefix_length, padding='pre')
        else:
            X = np.array(X, dtype=np.float32)

        y = np.array(y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        if encoding == "index":
            X_train = to_categorical(X_train, num_classes=len(activity_encoder.classes_))
            X_test = to_categorical(X_test, num_classes=len(activity_encoder.classes_))
            model = Sequential([
                Masking(mask_value=0., input_shape=(prefix_length, len(activity_encoder.classes_))),
                LSTM(64),
                Dense(1, activation='sigmoid')
            ])
        else:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob)

        results.append({
            'bucket': bucket,
            'encoding': encoding,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc': auc
        })


results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="bucket", y="f1_score", hue="encoding")
plt.title("F1 Score Comparison (LSTM on Sepsis Cases)")
plt.ylabel("F1 Score")
plt.xlabel("Bucketing Method")
plt.legend(title="Encoding")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
