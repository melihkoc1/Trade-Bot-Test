import pandas as pd
import numpy as np
import math
import os
import joblib

class SimulationManager:
    def __init__(self, simulations=1000, days=252):
        self.simulations = simulations
        self.days = days
        self.results_dir = "data/simulations"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def calculate_cagr(self, df):
        """Yıllık Bileşik Büyüme Oranı (CAGR)"""
        if df is None or len(df) < 10: return 0.05 # Varsayılan %5
        quote = df['Close']
        try:
            days = (quote.index[-1] - quote.index[0]).days
        except AttributeError:
            days = len(quote)
        if days <= 0: return 0.05
        # Negatif başlanğıç fiyatı kontrolü
        start_price = quote.iloc[0]
        if start_price <= 0: start_price = 0.01
        return ((((quote.iloc[-1]) / start_price)) ** (365.0/days)) - 1

    def calculate_volatility(self, df):
        """Yıllık Volatilite"""
        if df is None or len(df) < 10: return 0.20 # Varsayılan %20
        returns = df['Close'].pct_change()
        return returns.std() * np.sqrt(252)

    def run_monte_carlo(self, symbol, df, days_to_predict=21):
        """
        Monte Carlo Simülasyonu çalıştırır.
        """
        if df is None or df.empty:
            return None

        # Parametreleri hesapla
        mu = self.calculate_cagr(df)
        vol = self.calculate_volatility(df)
        start_price = df['Close'].iloc[-1]

        # Simülasyon Havuzu
        all_scenarios = np.zeros((self.simulations, days_to_predict + 1))
        all_scenarios[:, 0] = start_price

        # Vektörize edilmiş simülasyon (Daha hızlı)
        for t in range(1, days_to_predict + 1):
            # Geometrik Brownian Hareketi (GBM) Mantığı:
            # dS = S * (mu*dt + vol*epsilon*sqrt(dt))
            dt = 1 / 252
            shocks = np.random.normal(mu * dt, vol * np.sqrt(dt), self.simulations)
            all_scenarios[:, t] = all_scenarios[:, t-1] * (1 + shocks)

        # İstatistikleri hesapla
        final_prices = all_scenarios[:, -1]
        
        # VaR (Value at Risk) - %95 güvenle bir ayda (21 gün) beklenen max zarar
        var_95 = np.percentile(final_prices, 5)
        var_pct = (var_95 - start_price) / start_price * 100

        # Beklenen Getiri (Median)
        expected_price = np.median(final_prices)
        expected_return = (expected_price - start_price) / start_price * 100

        # Başarı Şansı (Fiyatın bugünkü fiyatın üstünde kalma olasılığı)
        success_prob = (final_prices > start_price).sum() / self.simulations * 100

        results = {
            "symbol": symbol,
            "start_price": round(start_price, 2),
            "expected_price": round(expected_price, 2),
            "expected_return_pct": round(expected_return, 2),
            "max_risk_pct": round(var_pct, 2), # VaR %95
            "success_probability": round(success_prob, 2),
            "percentile_5": round(np.percentile(final_prices, 5), 2),
            "percentile_95": round(np.percentile(final_prices, 95), 2),
            "scenarios": all_scenarios # Görselleştirme için tüm senaryolar
        }

        # Veriyi kaydet (Cache)
        # Sadece son 100 senaryoyu kaydedelim çok yer kaplamasın
        joblib.dump(results, os.path.join(self.results_dir, f"{symbol.replace('.IS','')}_sim.joblib"))
        
        return results

# Singleton
sim_engine = SimulationManager(simulations=1000)

def get_monte_carlo_results(symbol, df, days=21):
    return sim_engine.run_monte_carlo(symbol, df, days)
