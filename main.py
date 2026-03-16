from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Add this
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# --- ADD THIS SECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites to talk to your server
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------

model = joblib.load('crop_model.pkl')

class SoilData(BaseModel):
    n: float
    p: float
    k: float
    ph: float
    rain: float

@app.post("/predict")
def predict_crop(data: SoilData):
    input_features = np.array([[data.n, data.p, data.k, data.ph, data.rain]])
    prediction = model.predict(input_features)
    return {"recommended_crop": prediction[0]}