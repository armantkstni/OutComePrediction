import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "labeled_logs_csv_processed/BPIC17_O_Accepted.csv" 
LABEL_COL = "label"

df = pd.read_csv(DATA_PATH, sep=';')
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


class_counts = df[LABEL_COL].value_counts(normalize=False)
class_weights = df[LABEL_COL].value_counts(normalize=True)



plt.figure(figsize=(6,4))
class_counts.plot(kind='bar', color=['skyblue','salmon'])
plt.title("(Class Distribution)")
plt.xticks(rotation=0)
plt.show()
