import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import os

folder = r"C:\Users\Arman Takestani\Downloads\Compressed\predictive-monitoring-benchmark-master\predictive-monitoring-benchmark-master\labeled_logs_csv_processed"
print(" Files in folder:")
print(os.listdir(folder))


log_path = r"C:\Users\Arman Takestani\Downloads\Compressed\predictive-monitoring-benchmark-master\predictive-monitoring-benchmark-master\labeled_logs_csv_processed\Production.csv"
df = pd.read_csv(log_path)

X = df.drop(columns=["label"])  
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)


best_params = grid.best_params_

print(" Best params:", best_params)


out_path = r"C:\Users\Arman Takestani\Downloads\Compressed\predictive-monitoring-benchmark-master\best_param_pickles"
os.makedirs(out_path, exist_ok=True)

pickle_path = os.path.join(out_path, "optimal_params_rf_production_prefix_index.pickle")

with open(pickle_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📦 Best parameters saved to: {pickle_path}")
