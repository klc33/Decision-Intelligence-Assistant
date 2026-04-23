import time
import joblib
import pandas as pd
from backend.app.config import settings
from backend.app.utils.features import extract_features_single

class MLInferenceService:
    def __init__(self):
        model_pkg = joblib.load(settings.MODEL_PATH)
        self.pipeline = model_pkg["pipeline"]
        self.feature_names = model_pkg["feature_names"]
    
    def predict(self, text: str) -> dict:
        start = time.time()
        # Extract features as dict, convert to DataFrame with one row
        features_dict = extract_features_single(text)
        df = pd.DataFrame([features_dict])
        
        # Ensure correct column order (pipeline expects specific order)
        df = df[self.feature_names]
        
        # Predict
        proba = self.pipeline.predict_proba(df)[0]
        pred_class = self.pipeline.predict(df)[0]
        label = "urgent" if pred_class == 1 else "normal"
        confidence = max(proba)
        latency_ms = (time.time() - start) * 1000
        
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "cost_usd": 0.0  # free
        }