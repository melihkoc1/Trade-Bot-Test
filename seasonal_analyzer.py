import pandas as pd
import numpy as np
from datetime import datetime

class SeasonalAnalyzer:
    def __init__(self):
        self.lookback_years = 10

    def analyze_seasonality(self, df):
        """
        Hisse verisi üzerinden aylık getiri istatistiklerini hesaplar.
        """
        if df is None or len(df) < 252: # En az 1 yıllık veri lazım
            return None

        try:
            temp_df = df.copy()
            # Tarih indeksi olduğundan emin olalım
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                temp_df.index = pd.to_datetime(temp_df.index)

            # Aylık kapanışları al
            monthly_df = temp_df['Close'].resample('ME').last().to_frame()
            monthly_df['Return'] = monthly_df['Close'].pct_change() * 100
            monthly_df['Month'] = monthly_df.index.month
            monthly_df['Year'] = monthly_df.index.year
            
            # Son 10 yıla kısıtla
            current_year = datetime.now().year
            monthly_df = monthly_df[monthly_df['Year'] >= (current_year - self.lookback_years)]
            
            # Aylık bazda grupla
            seasonal_stats = monthly_df.groupby('Month')['Return'].agg([
                ('Avg_Return', 'mean'),
                ('Win_Rate', lambda x: (x > 0).mean() * 100),
                ('Count', 'count')
            ]).round(2)

            # Isı haritası verisi hazırlığı (Aylar vs Yıllar)
            heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Return').round(2)
            
            # Ay isimlerini ekle
            month_names = {
                1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 
                5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos", 
                9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
            }
            
            # İstatistikleri listeye çevir (UI için kolaylık)
            stats_list = []
            for m in range(1, 13):
                if m in seasonal_stats.index:
                    row = seasonal_stats.loc[m]
                    stats_list.append({
                        "month": m,
                        "month_name": month_names[m],
                        "avg_return": row['Avg_Return'],
                        "win_rate": row['Win_Rate'],
                        "count": int(row['Count'])
                    })

            return {
                "stats": stats_list,
                "heatmap": heatmap_data.to_dict(),
                "current_month": datetime.now().month
            }
        except Exception as e:
            print(f"Seasonal Analysis Error: {e}")
            return None

# Singleton
analyzer = SeasonalAnalyzer()

def get_seasonal_report(df):
    return analyzer.analyze_seasonality(df)

if __name__ == "__main__":
    # Test
    from data_fetcher import fetch_stock_data
    df = fetch_stock_data("THYAO.IS")
    res = get_seasonal_report(df)
    if res:
        print(f"THYAO Mevsimsel Analiz (Ay bazlı):")
        for s in res['stats']:
            print(f"- {s['month_name']}: Ort. Getiri %{s['avg_return']}, Başarı %{s['win_rate']}")
