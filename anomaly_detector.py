import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Klasör yapılandırması
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

DATA_PATH = "data/ml_dataset.csv"
MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "anomaly_scaler.joblib")

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['Log_Return', 'Volatility_10', 'RSI_14', 'Volume_Change', 'BB_Width']
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                print("[OK] Anomali Dedektoru (Isolation Forest) Yuklendi.")
            except Exception as e:
                print(f"[HATA] Anomali Modeli Yukleme Hatasi: {e}")

    def train(self):
        """
        Isolation Forest modelini eğitir. 
        Unsupervised learning olduğu için etiket (target) gerekmez.
        """
        if not os.path.exists(DATA_PATH):
            print(f"❌ Hata: {DATA_PATH} bulunamadı!")
            return

        print("📂 Anomali süzgeci için veri yükleniyor...")
        df = pd.read_csv(DATA_PATH, index_col=0)
        
        # Sadece ilgili özellikleri seç ve temizle
        X = df[self.features].copy()
        X = X.ffill().bfill()
        
        # Ölçeklendirme
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print("🧠 Isolation Forest eğitiliyor (Anomali oranı: %0.5)...")
        # contamination: Beklenen anomali oranı. %0.5 (binde 5) seçildi.
        self.model = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
        self.model.fit(X_scaled)
        
        # Kaydet
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"💾 Anomali modelleri '{MODEL_DIR}' klasörüne kaydedildi.")

    def is_anomaly(self, df):
        """
        Son verinin anomali olup olmadığını kontrol eder.
        Returns: True (Anomali var), False (Normal)
        """
        if self.model is None or self.scaler is None:
            return False, 0
            
        try:
            current_data = df[self.features].tail(1)
            X_scaled = self.scaler.transform(current_data)
            
            # predict returns 1 for normal, -1 for anomaly
            prediction = self.model.predict(X_scaled)[0]
            # score_samples returns anomaly score (lower means more abnormal)
            score = self.model.score_samples(X_scaled)[0]
            
            return (prediction == -1), round(float(score), 4)
        except Exception as e:
            print(f"Anomaly prediction error: {e}")
            return False, 0

# Singleton örneği
detector = AnomalyDetector()

def get_anomaly_status(df):
    """Strateji motoru için kolay erişim fonksiyonu"""
    return detector.is_anomaly(df)

if __name__ == "__main__":
    detector.train()
