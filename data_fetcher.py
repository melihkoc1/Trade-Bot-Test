import yfinance as yf
import pandas as pd
import time
from config import BIST50_SYMBOLS, DEFAULT_PERIOD, DEFAULT_INTERVAL, MACRO_SYMBOLS

# Temel analiz verileri çeyreklik güncellenir — 4 saatlik cache yeterli
_fundamental_cache = {}
_FUND_CACHE_TTL = 4 * 3600  # 4 saat

def fetch_stock_data(symbol, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """
    Belirtilen hisse sembolü için yfinance üzerinden fiyat verilerini çeker.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"[UYARI] {symbol} için veri çekilemedi. Belki sembol hatalı veya borsa o saatte verisi yok.")
            return None
            
        # Sadece bizim için gerekli olan temel OHLCV sütunlarını al
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # BIST Endekslerinde (ornegin XUTEK.IS) hacim bilgisi 0 donebilir.
        # Hacim bazli indikatorlerin (OBV, MFI) hesaplamada sonsuz veya NaN
        # uretip diger tum teknikleri (RSI vs) bozmamasi adina 0 hacimleri 1 yapiyoruz.
        df.loc[:, 'Volume'] = df['Volume'].replace(0, 1)
        
        # Sütun isimlerini bazen formüller hata vermesin diye küçük harfe veya istenen formata çevirebiliriz (opsiyonel)
        # Ancak yfinance standart olarak capital harfle verir.
        
        # Saat dilimini Türkiye'ye ayarla
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
        else:
            df.index = df.index.tz_convert('Europe/Istanbul')
            
        return df
        
    except Exception as e:
        print(f"[HATA] {symbol} verisi çekilirken hata oluştu: {e}")
        return None

def fetch_fundamental_data(symbol):
    """
    Belirtilen hisse icin temel analiz verilerini ceker.
    F/K, PD/DD, temettu verimi, piyasa degeri, EPS, sektor.
    Veriler çeyreklik güncellenir — 4 saatlik cache kullanılır.
    """
    global _fundamental_cache
    now = time.time()
    if symbol in _fundamental_cache:
        cached_time, cached_data = _fundamental_cache[symbol]
        if now - cached_time < _FUND_CACHE_TTL:
            return cached_data

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        data = {
            "pe_ratio": round(info.get("trailingPE", 0) or 0, 2),
            "forward_pe": round(info.get("forwardPE", 0) or 0, 2),
            "pb_ratio": round(info.get("priceToBook", 0) or 0, 2),
            "dividend_yield": round(info.get("dividendYield", 0) or 0, 2),
            "market_cap": info.get("marketCap", 0) or 0,
            "eps": round(info.get("trailingEps", 0) or 0, 2),
            "revenue": info.get("totalRevenue", 0) or 0,
            "sector": info.get("sector", "Bilinmiyor"),
            "beta": round(info.get("beta", 0) or 0, 2),
            "debt_to_equity": round(info.get("debtToEquity", 0) or 0, 2),
            "roe": round((info.get("returnOnEquity", 0) or 0) * 100, 2),
            "revenue_growth": round((info.get("revenueGrowth", 0) or 0) * 100, 2),
            "free_cashflow": info.get("freeCashflow", 0) or 0,
            "fifty_two_week_high": round(info.get("fiftyTwoWeekHigh", 0) or 0, 2),
            "fifty_two_week_low": round(info.get("fiftyTwoWeekLow", 0) or 0, 2),
            # --- ANALIST VERILERI (Elite Plus) ---
            "recommendation_key": info.get("recommendationKey", "Nötr"),
            "recommendation_mean": round(info.get("recommendationMean", 0) or 0, 2),
            "target_mean_price": round(info.get("targetMeanPrice", 0) or 0, 2),
            "target_low_price": round(info.get("targetLowPrice", 0) or 0, 2),
            "target_high_price": round(info.get("targetHighPrice", 0) or 0, 2),
            "number_of_analysts": info.get("numberOfAnalystOpinions", 0) or 0,
        }
        _fundamental_cache[symbol] = (now, data)
        return data
    except Exception as e:
        print(f"[HATA] {symbol} temel veri cekilemedi: {e}")
        return None

# Cache - makro verileri her taramada tekrar cekmemek icin
_macro_cache = {}

def fetch_macro_data(force_refresh=False):
    """
    Makro ekonomik verilerin son 1 aylik trendini ceker.
    Petrol, Altin, Dolar, BIST100
    """
    global _macro_cache
    if _macro_cache and not force_refresh:
        return _macro_cache
    
    result = {}
    for name, symbol in MACRO_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df is not None and len(df) >= 20:
                current = df["Close"].iloc[-1]
                one_month_ago = df["Close"].iloc[-22] if len(df) >= 22 else df["Close"].iloc[0]
                one_week_ago = df["Close"].iloc[-5] if len(df) >= 5 else df["Close"].iloc[0]
                
                monthly_change = round((current - one_month_ago) / one_month_ago * 100, 2)
                weekly_change = round((current - one_week_ago) / one_week_ago * 100, 2)
                
                if monthly_change > 3:
                    trend = "Yukari"
                elif monthly_change < -3:
                    trend = "Asagi"
                else:
                    trend = "Yatay"
                
                result[name] = {
                    "price": round(current, 2),
                    "monthly_change": monthly_change,
                    "weekly_change": weekly_change,
                    "trend": trend,
                }
            else:
                result[name] = {"price": 0, "monthly_change": 0, "weekly_change": 0, "trend": "Bilinmiyor"}
        except Exception as e:
            print(f"[HATA] Makro veri ({name}) cekilemedi: {e}")
            result[name] = {"price": 0, "monthly_change": 0, "weekly_change": 0, "trend": "Bilinmiyor"}
    
def fetch_macro_data_as_df(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """
    Makro verileri (BIST100, Dolar, Altin, Petrol) DataFrame formatinda ceker.
    """
    macro_dfs = {}
    for name, m_symbol in MACRO_SYMBOLS.items():
        try:
            m_ticker = yf.Ticker(m_symbol)
            m_df = m_ticker.history(period=period, interval=interval)
            if not m_df.empty:
                # TZ mismatch'i önlemek için lokalize et
                if m_df.index.tz is not None:
                    m_df.index = m_df.index.tz_localize(None)
                macro_dfs[name] = m_df[['Close']].rename(columns={'Close': f'Macro_{name}'})
        except Exception as e:
            print(f"[HATA] Makro DF ({name}) cekilemedi: {e}")
            
    return macro_dfs

def fetch_all_bist30(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """
    Tüm BIST30 hisselerinin verilerini çeker ve bir sözlükte (dictionary) saklar.
    """
    print(f"BIST30 verileri çekiliyor... (Süre: {period}, Periyot: {interval})")
    market_data = {}
    
    for symbol in BIST50_SYMBOLS:
        df = fetch_stock_data(symbol, period=period, interval=interval)
        if df is not None:
            market_data[symbol] = df
            
    print(f"Başarıyla veri çekilen hisse sayısı: {len(market_data)} / {len(BIST50_SYMBOLS)}")
    return market_data

if __name__ == "__main__":
    print("--- Veri Motoru Testi (Data Fetcher) ---")
    
    # 1. Tekil hisse test
    test_symbol = "THYAO.IS"
    print(f"\n[{test_symbol}] verisi getiriliyor...")
    thyao_verisi = fetch_stock_data(test_symbol)
    
    if thyao_verisi is not None:
        print(f"\n{test_symbol} Son 5 Verilik Tablo:")
        print(thyao_verisi.tail())
    else:
        print(f"{test_symbol} verisi alınamadı.")
        
    # 2. Toplu Data çekimini test etmek istersek açabiliriz
    # all_data = fetch_all_bist30()
    # print(all_data.keys())
