import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import joblib

class MarketClusterer:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model_path = "models/market_clusters.joblib"
        self.scaler_path = "models/cluster_scaler.joblib"
        self.features = ['Volatility_10', 'Log_Return', 'RSI_14', 'Volume_Change']

    def train_clusters(self, market_data):
        """
        Tüm BIST50 hisselerinin son durumuna göre kümeleme yapar.
        market_data: {symbol: df} sözlüğü
        """
        summary_list = []
        for symbol, df in market_data.items():
            if df is not None and len(df) > 20:
                last = df[self.features].tail(1).copy()
                last['Symbol'] = symbol
                summary_list.append(last)
        
        if not summary_list:
            return None
            
        full_df = pd.concat(summary_list).reset_index(drop=True)
        X = full_df[self.features]
        
        # Ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        full_df['Cluster'] = kmeans.fit_transform(X_scaled).argmin(axis=1) # Basitleştirmek için labels_
        full_df['Cluster'] = kmeans.labels_
        
        # Küme isimlerini karakterize et (Basit mantık)
        cluster_info = {}
        for i in range(self.n_clusters):
            c_data = full_df[full_df['Cluster'] == i]
            avg_vol = c_data['Volatility_10'].mean()
            avg_ret = c_data['Log_Return'].mean()
            
            if avg_vol > full_df['Volatility_10'].mean() * 1.2:
                name = "Yüksek Volatilite (Agresif)"
            elif avg_ret > 0.01:
                name = "Güçlü Trend (Liderler)"
            elif avg_vol < full_df['Volatility_10'].mean() * 0.8:
                name = "Düşük Volatiliteli (Defansif)"
            else:
                name = "Stabil / Yatay"
            cluster_info[i] = name

        # Kaydet
        joblib.dump(kmeans, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        joblib.dump(cluster_info, "models/cluster_names.joblib")
        
        return full_df[['Symbol', 'Cluster']]

    def get_market_map(self, market_data):
        """Arayüz için tüm piyasa haritasını döner"""
        clusters = self.train_clusters(market_data)
        if clusters is None:
            return None
            
        cluster_names = joblib.load("models/cluster_names.joblib")
        clusters['Cluster_Name'] = clusters['Cluster'].map(cluster_names)
        return clusters

# Singleton
clusterer = MarketClusterer()

def get_market_clusters(market_data):
    return clusterer.get_market_map(market_data)
