"""
to run the app 
go to root project folder and run the code below
    uvicorn src.app:app --reload
and then go to  address  :   http://127.0.0.1:8000/docs

"""



import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# پیدا کردن مسیر دقیق فایل مدل
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_xgb_model.pkl")

# بارگذاری مدل با مدیریت خطا
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"مدل در مسیر زیر پیدا نشد: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Fraud Detection API", description="Real-time Credit Card Fraud Detection")

# ساختار دیتای ورودی
class Transaction(BaseModel):
    # فرض بر این است که ۲۹ ویژگی (V1-V28 + Log_Amount) را می‌فرستیم
    features: list 

@app.get("/")
def home():
    return {"message": "Fraud Detection System is Online!"}

@app.post("/predict")
def predict_fraud(data: Transaction):
    try:
        # تبدیل لیست ورودی به آرایه دو بعدی برای مدل
        input_data = np.array(data.features).reshape(1, -1)
        
        # چک کردن تعداد ویژگی‌ها (باید ۲۹ تا باشد)
        if input_data.shape[1] != 29:
            raise HTTPException(status_code=400, detail=f"تعداد ویژگی‌ها باید ۲۹ باشد، اما {input_data.shape[1]} دریافت شد.")

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "prediction": "Fraud" if int(prediction[0]) == 1 else "Normal",
            "fraud_probability": f"{probability:.4f}",
            "is_fraud": int(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))