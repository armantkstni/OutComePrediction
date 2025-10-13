import pandas as pd
import matplotlib.pyplot as plt

# مسیر فایل داده‌هات
DATA_PATH = "labeled_logs_csv_processed/BPIC17_O_Accepted.csv"  # تغییر بده اگر فایلت چیز دیگه‌ست
LABEL_COL = "label"  # ستون برچسب (همون که توی pipeline استفاده کردی)

# 1. خواندن داده
df = pd.read_csv(DATA_PATH, sep=';')
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 2. شمارش هر کلاس
class_counts = df[LABEL_COL].value_counts(normalize=False)
class_weights = df[LABEL_COL].value_counts(normalize=True)

print("📊 تعداد نمونه‌ها در هر کلاس:")
print(class_counts)
print("\n⚖️ نسبت هر کلاس (وزن):")
print(class_weights)

# 3. رسم نمودار
plt.figure(figsize=(6,4))
class_counts.plot(kind='bar', color=['skyblue','salmon'])
plt.title("توزیع کلاس‌ها (Class Distribution)")
plt.xlabel("کلاس")
plt.ylabel("تعداد نمونه")
plt.xticks(rotation=0)
plt.show()
