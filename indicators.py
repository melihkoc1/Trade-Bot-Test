import pandas as pd
import numpy as np

# ============================================================
# TEMEL İNDİKATÖR HESAPLAMALARI
# ============================================================

def calculate_sma(series, length):
    return series.rolling(window=length).mean()

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_rsi(series, length=14):
    """
    RSI hesaplaması — TradingView ile birebir uyumlu.
    Wilder's Smoothing (RMA) kullanır, SMA değil.
    RMA = EMA(alpha=1/length) = ewm(alpha=1/length)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's Smoothing (RMA): alpha = 1/length
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = calculate_ema(series, fast)
    exp2 = calculate_ema(series, slow)
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator (%K ve %D)"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_obv(close, volume):
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv

def calculate_momentum(series, length=10):
    """Momentum"""
    return series - series.shift(length)


def calculate_adx(high, low, close, length=14):
    """
    ADX (Average Directional Index) — TradingView uyumlu.
    Wilder's Smoothing (RMA) kullanir.
    Returns: adx, plus_di, minus_di
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM ve -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    
    # Wilder's Smoothing (RMA)
    atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    # +DI ve -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX ve ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, length=14):
    """
    ATR (Average True Range) — Volatilite olcumu.
    Wilder's Smoothing (RMA) ile TradingView uyumlu.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    return atr

def find_support_resistance(df, window=20, num_levels=3):
    """
    Destek ve direnc seviyelerini tespit eder.
    Pivot tabanlı: Yerel dip = destek, yerel tepe = direnc.
    En yakin 'num_levels' adet seviye doner.
    """
    supports = []
    resistances = []
    
    highs = df["High"].values
    lows = df["Low"].values
    close_now = df["Close"].iloc[-1]
    
    half = window // 2
    
    for i in range(half, len(df) - half):
        # Yerel dip (destek)
        if lows[i] == min(lows[i-half:i+half+1]):
            supports.append(round(lows[i], 2))
        # Yerel tepe (direnc)
        if highs[i] == max(highs[i-half:i+half+1]):
            resistances.append(round(highs[i], 2))
    
    # Benzersizlestir ve fiyata gore sirala
    supports = sorted(set(supports))
    resistances = sorted(set(resistances))
    
    # Mevcut fiyatin ALTINDAKI en yakin destekler
    nearby_supports = [s for s in supports if s < close_now]
    nearby_supports = nearby_supports[-num_levels:] if nearby_supports else []
    
    # Mevcut fiyatin USTUNDEKI en yakin direncler
    nearby_resistances = [r for r in resistances if r > close_now]
    nearby_resistances = nearby_resistances[:num_levels] if nearby_resistances else []
    
    return {
        "supports": nearby_supports,
        "resistances": nearby_resistances,
        "nearest_support": nearby_supports[-1] if nearby_supports else None,
        "nearest_resistance": nearby_resistances[0] if nearby_resistances else None,
        "support_distance_pct": round((close_now - nearby_supports[-1]) / close_now * 100, 2) if nearby_supports else None,
        "resistance_distance_pct": round((nearby_resistances[0] - close_now) / close_now * 100, 2) if nearby_resistances else None,
    }

def calculate_ema(series, length):
    """EMA (Exponential Moving Average)"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_fibonacci(df, lookback=50):
    """
    Fibonacci Retracement seviyeleri.
    Son 'lookback' mumda en yuksek ve en dusuk noktayi bulur,
    onlara gore %23.6, %38.2, %50, %61.8 seviyelerini hesaplar.
    """
    recent = df.tail(lookback)
    high = recent["High"].max()
    low = recent["Low"].min()
    close = df["Close"].iloc[-1]
    diff = high - low
    
    if diff == 0:
        return None
    
    # Dusus trendi varsayimi (yuksekten dusuge)
    fib_levels = {
        "high": round(high, 2),
        "low": round(low, 2),
        "fib_236": round(high - diff * 0.236, 2),
        "fib_382": round(high - diff * 0.382, 2),
        "fib_500": round(high - diff * 0.500, 2),
        "fib_618": round(high - diff * 0.618, 2),
    }
    
    # Fiyat hangi Fibonacci bolgesi icinde?
    if close >= fib_levels["fib_236"]:
        fib_levels["zone"] = "Tepe bolge (0-23.6%)"
        fib_levels["zone_score"] = -2  # Pahali
    elif close >= fib_levels["fib_382"]:
        fib_levels["zone"] = "23.6-38.2% duzeltme"
        fib_levels["zone_score"] = 1  # Hafif ucuz
    elif close >= fib_levels["fib_500"]:
        fib_levels["zone"] = "38.2-50% duzeltme (ideal giris)"
        fib_levels["zone_score"] = 3  # Iyi giris
    elif close >= fib_levels["fib_618"]:
        fib_levels["zone"] = "50-61.8% duzeltme (guclu giris)"
        fib_levels["zone_score"] = 5  # Cok iyi giris
    else:
        fib_levels["zone"] = "61.8%+ derin duzeltme"
        fib_levels["zone_score"] = 4  # Derin ama riskli
    
    return fib_levels


def check_minervini_template(df):
    """
    Mark Minervini'nin "Trend Template" (8 Kural) kontrolü.
    Gerçek bir ralli başlangıcını tespit etmek için kullanılır.
    """
    if len(df) < 200: return False, 0
    
    last = df.iloc[-1]
    low_52w = df['Low'].rolling(window=250).min().iloc[-1]
    high_52w = df['High'].rolling(window=250).max().iloc[-1]
    
    # Kurallar (Sütun kontrolü ekleyerek güvenli hale getirildi)
    try:
        c1 = last['Close'] > last['SMA_150'] and last['Close'] > last['SMA_200']
        c2 = last['SMA_150'] > last['SMA_200']
        c3 = last['SMA_200'] > df['SMA_200'].shift(20).iloc[-1] 
        c4 = last['SMA_50'] > last['SMA_150'] and last['SMA_50'] > last['SMA_200']
        c5 = last['Close'] > last['SMA_50']
        c6 = last['Close'] > (low_52w * 1.30) 
        c7 = last['Close'] > (high_52w * 0.75) 
        
        # BIST kalibrasyonu: TL enflasyonu nedeniyle %20 yıllık getiri eşiği anlamsız.
        # %30 daha anlamlı bir momentum eşiği.
        c8 = (last['Close'] / df['Close'].shift(250).iloc[-1] > 1.30)
        
        rules = [c1, c2, c3, c4, c5, c6, c7, c8]
        score = sum(rules)
        return (score >= 7), score
    except KeyError as e:
        print(f"Minervini Template Error: {e} missing")
        return False, 0
    
    # Canlı veri akışında RS Rating (Görece Güç) için basit bir vekil:
    # BIST kalibrasyonu: %30 yıllık getiri eşiği (TL enflasyonu için daha anlamlı)
    c8 = (last['Close'] / df['Close'].shift(250).iloc[-1] > 1.30)
    
    rules = [c1, c2, c3, c4, c5, c6, c7, c8]
    score = sum(rules)
    return (score >= 7), score

def check_green_line_breakout(df):
    """
    Green Line (Tüm zamanların veya yıllık büyük zirve) tespiti.
    Zirveye yakınlık ve kırılım durumunu döner.
    """
    if len(df) < 60: return False, 0, 0
    
    # Aylık en yüksekleri bul (GLV mantığı)
    monthly_highs = df.resample('ME')['High'].max()
    last_glv = 0
    for h in monthly_highs:
        if h > last_glv:
            last_glv = h
            
    current_price = df['Close'].iloc[-1]
    diff_pct = (last_glv - current_price) / current_price * 100
    
    # Eğer fiyat zirvenin %5 içindeyse veya üzerindeyse
    is_near = abs(diff_pct) <= 5 or current_price >= last_glv
    return is_near, round(diff_pct, 2), round(last_glv, 2)


def calculate_supertrend(df, length=7, multiplier=3):
    """
    SuperTrend İndikatörü hesaplaması.
    Trendin yönünü (AL/SAT) ve volatiliteli stop-loss seviyesini belirler.
    """
    # ATR hesapla (Zaten ATR_14 vardır ama SuperTrend için özel periyot gerekebilir)
    # TradingView standardı: hl2 + mult * atr
    hl2 = (df['High'] + df['Low']) / 2
    atr = calculate_atr(df['High'], df['Low'], df['Close'], length)
    
    # Üst ve Alt bantlar
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Final bantları hesapla (Trailing mantığı)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int) # 1 Bull, -1 Bear
    
    for i in range(1, len(df)):
        # Final Upper
        if df['Close'].iloc[i-1] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]
            
        # Final Lower
        if df['Close'].iloc[i-1] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]
            
        # SuperTrend Yönü
        if i == 1:
            supertrend.iloc[i] = final_upper.iloc[i]
            direction.iloc[i] = -1
        else:
            if supertrend.iloc[i-1] == final_upper.iloc[i-1] and df['Close'].iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and df['Close'].iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = direction.iloc[i-1]
                supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]
                
    return supertrend, direction


# ============================================================
# TÜM İNDİKATÖRLERİ DATAFRAME'E EKLE
# ============================================================

def add_technical_indicators(df):
    """
    Verilen DataFrame üzerine teknik indikatörleri manuel olarak ekler.
    Divergence tespiti için gerekli tüm indikatörleri hesaplar.
    """
    if df is None or df.empty:
        return None
    
    # 1. Hareketli Ortalamalar (SMA)
    df["SMA_9"] = calculate_sma(df["Close"], 9)
    df["SMA_21"] = calculate_sma(df["Close"], 21)
    df["SMA_50"] = calculate_sma(df["Close"], 50)
    df["SMA_150"] = calculate_sma(df["Close"], 150)
    df["SMA_200"] = calculate_sma(df["Close"], 200)
    
    # 2. RSI
    df["RSI_14"] = calculate_rsi(df["Close"], 14)
    
    # 3. MACD
    macd, signal, hist = calculate_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    
    # 4. Bollinger Bantları
    sma_20 = calculate_sma(df["Close"], 20)
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = sma_20 + (std_20 * 2)
    df["BB_Lower"] = sma_20 - (std_20 * 2)
    df["BB_Middle"] = sma_20
    
    # 5. Stochastic (Divergence için)
    stoch_k, stoch_d = calculate_stochastic(df["High"], df["Low"], df["Close"])
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_d
    
    # 6. OBV (Divergence için)
    df["OBV"] = calculate_obv(df["Close"], df["Volume"])
    
    # 7. Momentum (Divergence için)
    df["MOM_10"] = calculate_momentum(df["Close"], 10)
    
    # 9. ADX (Trend gucu)
    adx, plus_di, minus_di = calculate_adx(df["High"], df["Low"], df["Close"], 14)
    df["ADX"] = adx
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    
    # 10. ATR (Volatilite)
    df["ATR_14"] = calculate_atr(df["High"], df["Low"], df["Close"], 14)
    
    # 11. EMA (Trend takibi için temel EMAlar)
    df["EMA_9"] = calculate_ema(df["Close"], 9)
    df["EMA_21"] = calculate_ema(df["Close"], 21)
    df["EMA_50"] = calculate_ema(df["Close"], 50)

    # 12. Minervini & Green Line (Elite Plus)
    # Bu hesaplamalar SMA 200 ve 52 haftalık veriye dayandığı için en sonda hesaplanır
    is_minervini, m_score = check_minervini_template(df)
    df["MINERVINI_SCORE"] = m_score
    
    is_glv, glv_diff, glv_val = check_green_line_breakout(df)
    df["GLV_NEAR"] = is_glv
    df["GLV_DIFF"] = glv_diff
    df["GLV_VAL"] = glv_val

    # 13. SuperTrend
    st_line, st_dir = calculate_supertrend(df)
    df["SUPERTREND"] = st_line
    df["SUPERTREND_DIR"] = st_dir # 1 Bull, -1 Bear
    
    # --- ML ODAKLI ÖZELLİKLER (MODELİN DAHA İYİ ÖĞRENMESİ İÇİN) ---
    # Log-Returns (Fiyat değişimlerini daha stabil temsil eder)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Volatilite (Belirli bir penceredeki standart sapma)
    df["Volatility_10"] = df["Log_Return"].rolling(window=10).std()
    
    # Fiyatın hareketli ortalamalara uzaklığı (Trend yönü ve gücü için)
    df["Dist_SMA_50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
    df["Dist_SMA_200"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"]
    
    # RSI Eğimi (Momentumun hızlanıp yavaşladığını anlamak için)
    df["RSI_Slope"] = df["RSI_14"].diff(3)
    
    # Hacim Değişimi
    df["Volume_Change"] = df["Volume"].pct_change()
    
    # Bollinger Bant Genişliği (Squeeze tespiti için)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    
    # Mum yapısı (Gövde büyüklüğü ve fitiller)
    df["Body_Size"] = (df["Close"] - df["Open"]).abs() / df["Open"]
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]
    
    # Makro Bazlı Log-Returns (Eğer varsa)
    macro_cols = [c for c in df.columns if c.startswith("Macro_")]
    for col in macro_cols:
        df[f"{col}_LogRet"] = np.log(df[col] / df[col].shift(1))
                
    return df

# ============================================================
# DIVERGENCE (UYUMSUZLUK) TESPİTİ
# Pine Script mantığının Python'a çevrilmiş hali
# ============================================================

def find_pivot_highs(series, left_bars=5, right_bars=5):
    """Pivot High (tepe) noktalarını bulur. TradingView'daki pivothigh() fonksiyonunun karşılığı."""
    pivots = pd.Series(np.nan, index=series.index)
    for i in range(left_bars, len(series) - right_bars):
        window_left = series.iloc[i - left_bars : i]
        window_right = series.iloc[i + 1 : i + right_bars + 1]
        if series.iloc[i] > window_left.max() and series.iloc[i] > window_right.max():
            pivots.iloc[i] = series.iloc[i]
    return pivots

def find_pivot_lows(series, left_bars=5, right_bars=5):
    """Pivot Low (dip) noktalarını bulur. TradingView'daki pivotlow() fonksiyonunun karşılığı."""
    pivots = pd.Series(np.nan, index=series.index)
    for i in range(left_bars, len(series) - right_bars):
        window_left = series.iloc[i - left_bars : i]
        window_right = series.iloc[i + 1 : i + right_bars + 1]
        if series.iloc[i] < window_left.min() and series.iloc[i] < window_right.min():
            pivots.iloc[i] = series.iloc[i]
    return pivots

def check_no_cutthrough(indicator_series, start_idx, end_idx, direction="positive"):
    """
    Pine Script'teki nocut1/nocut2 fonksiyonunun karşılığı.
    İki pivot arasında çizilen çizgiyi indikatörün kesip kesmediğini kontrol eder.
    """
    if start_idx >= end_idx or end_idx >= len(indicator_series):
        return False
    
    start_val = indicator_series.iloc[start_idx]
    end_val = indicator_series.iloc[end_idx]
    length = end_idx - start_idx
    
    if length <= 1:
        return True
    
    diff = (end_val - start_val) / length
    
    for x in range(1, length):
        line_val = start_val + diff * x
        actual_val = indicator_series.iloc[start_idx + x]
        
        if pd.isna(actual_val):
            continue
        
        if direction == "positive" and actual_val < line_val:
            return False  # Çizgiyi aşağı kesti
        elif direction == "negative" and actual_val > line_val:
            return False  # Çizgiyi yukarı kesti
    
    return True

def detect_divergences(df, left_bars=5, right_bars=5, check_cutthrough=True):
    """
    TradingView "Divergence for many indicators v3" mantığıyla
    birden fazla indikatörde uyumsuzluk tespit eder.
    
    Returns:
        DataFrame'e bullish_div_count, bearish_div_count ve detay sütunları eklenir.
    """
    if df is None or len(df) < (left_bars + right_bars + 10):
        return df
    
    # Pivot noktalarını bul
    pivot_highs = find_pivot_highs(df["High"], left_bars, right_bars)
    pivot_lows = find_pivot_lows(df["Low"], left_bars, right_bars)
    
    # Kontrol edilecek indikatörler (Pine Script'teki 11 indikatörden en güçlü 5'i + MACD Hist)
    indicators = {
        "RSI": "RSI_14",
        "MACD": "MACD",
        "MACD_Hist": "MACD_Hist",
        "Stoch": "STOCH_K",
        "OBV": "OBV",
        "MOM": "MOM_10",
    }
    
    # Sonuç sütunları
    df["bull_div_count"] = 0
    df["bear_div_count"] = 0
    df["bull_div_details"] = ""
    df["bear_div_details"] = ""
    
    # Son pivot indekslerini takip et
    last_pivot_high_idx = None
    last_pivot_low_idx = None
    
    for i in range(len(df)):
        # Pivot high bulundu mu?
        if not pd.isna(pivot_highs.iloc[i]):
            # Eğer önceki bir pivot high varsa, bearish divergence ara
            if last_pivot_high_idx is not None:
                price_now = df["High"].iloc[i]
                price_prev = df["High"].iloc[last_pivot_high_idx]
                
                # Fiyat yeni tepe yaptı AMA indikatörler yapmadı = Bearish Divergence
                if price_now > price_prev:
                    bear_count = 0
                    bear_names = []
                    
                    for name, col in indicators.items():
                        if col not in df.columns:
                            continue
                        indi_now = df[col].iloc[i]
                        indi_prev = df[col].iloc[last_pivot_high_idx]
                        
                        if pd.isna(indi_now) or pd.isna(indi_prev):
                            continue
                        
                        # İndikatör düştü ama fiyat yükseldi = Negatif uyumsuzluk
                        if indi_now < indi_prev:
                            if not check_cutthrough or check_no_cutthrough(
                                df[col], last_pivot_high_idx, i, "negative"
                            ):
                                bear_count += 1
                                bear_names.append(name)
                    
                    if bear_count > 0:
                        df.iloc[i, df.columns.get_loc("bear_div_count")] = bear_count
                        df.iloc[i, df.columns.get_loc("bear_div_details")] = ", ".join(bear_names)
            
            last_pivot_high_idx = i
        
        # Pivot low bulundu mu?
        if not pd.isna(pivot_lows.iloc[i]):
            # Eğer önceki bir pivot low varsa, bullish divergence ara
            if last_pivot_low_idx is not None:
                price_now = df["Low"].iloc[i]
                price_prev = df["Low"].iloc[last_pivot_low_idx]
                
                # Fiyat yeni dip yaptı AMA indikatörler yeni dip yapmadı = Bullish Divergence
                if price_now < price_prev:
                    bull_count = 0
                    bull_names = []
                    
                    for name, col in indicators.items():
                        if col not in df.columns:
                            continue
                        indi_now = df[col].iloc[i]
                        indi_prev = df[col].iloc[last_pivot_low_idx]
                        
                        if pd.isna(indi_now) or pd.isna(indi_prev):
                            continue
                        
                        # İndikatör yükseldi ama fiyat düştü = Pozitif uyumsuzluk
                        if indi_now > indi_prev:
                            if not check_cutthrough or check_no_cutthrough(
                                df[col], last_pivot_low_idx, i, "positive"
                            ):
                                bull_count += 1
                                bull_names.append(name)
                    
                    if bull_count > 0:
                        df.iloc[i, df.columns.get_loc("bull_div_count")] = bull_count
                        df.iloc[i, df.columns.get_loc("bull_div_details")] = ", ".join(bull_names)
            
            last_pivot_low_idx = i
    
    return df

def get_recent_divergence(df, lookback=10):
    """
    Son N mum içindeki en güçlü divergence bilgisini döner.
    Strategy ve Dashboard tarafından kullanılır.
    """
    if df is None or "bull_div_count" not in df.columns:
        return {"bullish": 0, "bearish": 0, "bull_details": "", "bear_details": ""}
    
    recent = df.tail(lookback)
    
    bull_max_idx = recent["bull_div_count"].idxmax() if recent["bull_div_count"].max() > 0 else None
    bear_max_idx = recent["bear_div_count"].idxmax() if recent["bear_div_count"].max() > 0 else None
    
    result = {
        "bullish": int(recent["bull_div_count"].max()) if bull_max_idx else 0,
        "bearish": int(recent["bear_div_count"].max()) if bear_max_idx else 0,
        "bull_details": recent.loc[bull_max_idx, "bull_div_details"] if bull_max_idx else "",
        "bear_details": recent.loc[bear_max_idx, "bear_div_details"] if bear_max_idx else "",
    }
    return result


# ============================================================
# TEST BLOĞU
# ============================================================
if __name__ == "__main__":
    from data_fetcher import fetch_stock_data
    
    print("=" * 55)
    print("  DIVERGENCE (UYUMSUZLUK) TESTİ")
    print("=" * 55)
    
    test_symbol = "THYAO.IS"
    print(f"\n[{test_symbol}] verisi ve indikatörleri hesaplanıyor...")
    df = fetch_stock_data(test_symbol, period="1y", interval="1d")
    
    if df is not None:
        df = add_technical_indicators(df)
        df = detect_divergences(df, left_bars=5, right_bars=5)
        
        # Bulunan uyumsuzlukları göster
        bull_divs = df[df["bull_div_count"] > 0][["Close", "RSI_14", "bull_div_count", "bull_div_details"]]
        bear_divs = df[df["bear_div_count"] > 0][["Close", "RSI_14", "bear_div_count", "bear_div_details"]]
        
        print(f"\nBullish (Yukari) Uyumsuzluklar ({len(bull_divs)} adet):")
        if len(bull_divs) > 0:
            for idx, row in bull_divs.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                print(f"  {date_str} | Fiyat: {row['Close']:.2f} | RSI: {row['RSI_14']:.1f} | "
                      f"Uyumsuz: {int(row['bull_div_count'])} ({row['bull_div_details']})")
        else:
            print("  Bulunamadi.")
        
        print(f"\nBearish (Asagi) Uyumsuzluklar ({len(bear_divs)} adet):")
        if len(bear_divs) > 0:
            for idx, row in bear_divs.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                print(f"  {date_str} | Fiyat: {row['Close']:.2f} | RSI: {row['RSI_14']:.1f} | "
                      f"Uyumsuz: {int(row['bear_div_count'])} ({row['bear_div_details']})")
        else:
            print("  Bulunamadi.")
        
        # Son durum
        div = get_recent_divergence(df, lookback=15)
        print(f"\nSon 15 Mum Uyumsuzluk Durumu:")
        print(f"  Bullish: {div['bullish']} indikatör ({div['bull_details']})")
        print(f"  Bearish: {div['bearish']} indikatör ({div['bear_details']})")
    else:
        print("Veri cekilemedi.")
