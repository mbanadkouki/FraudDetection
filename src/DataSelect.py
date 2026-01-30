import pandas as pd
import numpy as np
import os
import json
# بارگذاری دیتای اصلی
current_dir = os.getcwd()

file_path = os.path.join(current_dir, "data", "creditcard.csv")

df_full = pd.read_csv(file_path)

# پیدا کردن اولین ردیف تقلب
fraud_sample = df_full[df_full['Class'] == 1].iloc[0]

# استخراج ویژگی‌ها و محاسبه Log_Amount
features = fraud_sample.iloc[1:29].tolist()
log_amount_val = float(np.log1p(fraud_sample['Amount']))
features.append(log_amount_val)

print("Fraud JSON for testing:")
print(json.dumps({"features": features}))