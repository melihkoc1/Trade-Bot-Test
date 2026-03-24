import pandas as pd
import numpy as np
import os
import joblib

class StrategyOptimizer:
    def __init__(self):
        self.strategies = ["RSI_Reversal", "MACD_Cross", "SMA_Cross", "BB_Lower_Touch"]
        self.results_cache = {}
        self.cache_path = "models/champion_indicators.joblib"
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                self.results_cache = joblib.load(self.cache_path)
            except:
                self.results_cache = {}

    def find_champion_indicator(self, symbol, df):
        """
        Hisse için son 1 yılda en başarılı indikatörü bulur.
        """
        if df is None or len(df) < 100:
            return None

        # Sadece Kapanış fiyatı üzerinden hızlı testler
        data = df.copy()
        
        # Strateji 1: RSI (30'dan yukarı dönüş)
        data['RSI_Signal'] = (data['RSI_14'] > 30) & (data['RSI_14'].shift(1) <= 30)
        
        # Strateji 2: MACD (Golden Cross)
        data['MACD_Signal'] = (data['MACD_Hist'] > 0) & (data['MACD_Hist'].shift(1) <= 0)
        
        # Strateji 3: SMA (9/21 Golden Cross)
        data['SMA_Signal'] = (data['SMA_9'] > data['SMA_21']) & (data['SMA_9'].shift(1) <= data['SMA_21'].shift(1))
        
        # Strateji 4: Bollinger (Alt Band temasından sonra toparlanma)
        data['BB_Signal'] = (data['Close'] > data['BB_Lower']) & (data['Close'].shift(1) <= data['BB_Lower'].shift(1))

        # Performans ölçümü (5 gün sonraki getiri)
        data['Return_5d'] = (data['Close'].shift(-5) / data['Close'] - 1) * 100
        
        results = {}
        for strat in ["RSI_Signal", "MACD_Signal", "SMA_Signal", "BB_Signal"]:
            trades = data[data[strat] == True]
            if len(trades) > 3:
                win_rate = (trades['Return_5d'] > 0).mean() * 100
                avg_ret = trades['Return_5d'].mean()
                results[strat.replace("_Signal", "")] = win_rate + (avg_ret * 2) # Puanlama formülü

        if not results:
            return None

        champion = max(results, key=results.get)
        
        self.results_cache[symbol] = {
            "name": champion,
            "win_rate": round(results[champion], 1)
        }
        
        # Önbelleği güncelle
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.results_cache, self.cache_path)
        
        return self.results_cache[symbol]

    def get_champion_indicator(self, symbol, df):
        if symbol in self.results_cache:
            return self.results_cache[symbol]
        return self.find_champion_indicator(symbol, df)

# Singleton
optimizer = StrategyOptimizer()

def get_champion(symbol, df):
    return optimizer.get_champion_indicator(symbol, df)

if __name__ == "__main__":
    # Test
    from data_fetcher import fetch_stock_data
    from indicators import add_technical_indicators
    df = fetch_stock_data("THYAO.IS")
    df = add_technical_indicators(df)
    res = get_champion("THYAO.IS", df)
    print(f"THYAO Şampiyon İndikatörü: {res}")
