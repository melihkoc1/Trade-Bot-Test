from data_fetcher import fetch_stock_data, fetch_fundamental_data, fetch_macro_data
from indicators import add_technical_indicators, detect_divergences, get_recent_divergence, find_support_resistance, calculate_fibonacci
from config import SECTOR_MAP, SECTOR_MACRO_CORRELATION
import anomaly_detector
import sim_manager
import news_scraper
import catalyst_manager
import seasonal_analyzer
import optimizer

# ============================================================
# YATIRIM PROFİLLERİ
# ============================================================
PROFILES = {
    "Trend Avcisi": {
        "period": "6mo",
        "interval": "1d",
        "description": "2 hafta - 3 ay vadeli. Trend baslangiclarini yakala, %10-50 kar hedefi.",
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "stop_loss": -7,
        "take_profit": 10,
    },
    "Deger Yatirimcisi": {
        "period": "2y",
        "interval": "1wk",
        "description": "Uzun vadeli. Dusuk fiyattan al, sabret, degeri gelince sat.",
        "rsi_oversold": 30,
        "rsi_overbought": 75,
        "stop_loss": -15,
        "take_profit": 30,
    },
    "Manuel": {
        "period": "6mo",
        "interval": "1d",
        "description": "Kendi ayarlarini sec.",
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "stop_loss": -7,
        "take_profit": 10,
    },
}


# ============================================================
# MODÜLER STRATEJİ FONKSİYONLARI
# Her modül (ham_skor, oy, gerekçeler) döner.
# ham_skor: -4..+4  |  oy: 'AL' / 'BEKLE' / 'SAT'
# ============================================================

def _module_trend(result):
    """Trend modülü: net bir yukari trend var mi?"""
    score, reasons = 0, []
    price = result.get("price")

    # Yapisal trend filtresi: SMA200 varsa kullan, yoksa SMA50 ile devam et
    # Amac: yapısal düşüş trendindeki hisselere (KONTR, SMRTG gibi) girmemek
    sma200 = result.get("sma_200")
    sma50  = result.get("sma_50")
    trend_ma = sma200 if sma200 else sma50
    trend_ma_label = "SMA200" if sma200 else "SMA50"
    if trend_ma and price:
        if price > trend_ma:
            score += 1; reasons.append(f"[Trend] Fiyat {trend_ma_label} uzerinde — yapisal yukselis")
        elif price < trend_ma * 0.95:
            score -= 2; reasons.append(f"[Trend] Fiyat {trend_ma_label}'un %5+ altinda — yapisal dusus")
        else:
            score -= 1; reasons.append(f"[Trend] Fiyat {trend_ma_label} altinda — zayif yapi")

    # SMA 9/21 — kisa vadeli fiyat konumu
    sma9, sma21 = result.get("sma_9"), result.get("sma_21")
    if sma9 and sma21 and price:
        if sma9 > sma21 and price > sma21:
            score += 1; reasons.append("[Trend] SMA: Fiyat ve kisa ort. yukarda")
        elif sma9 < sma21 and price < sma21:
            score -= 1; reasons.append("[Trend] SMA: Fiyat ve kisa ort. asagida")

    # ADX + trend yonu
    adx, td = result.get("adx"), result.get("trend_direction")
    if adx is not None:
        if adx >= 25:
            if td == "Yukari":
                score += 1; reasons.append(f"[Trend] ADX guclu yukari trend ({adx})")
            elif td == "Asagi":
                score -= 1; reasons.append(f"[Trend] ADX guclu asagi trend ({adx})")
        # ADX < 20: trendsiz — ne odul ne ceza

    # SuperTrend yonu
    st = result.get("supertrend_dir", 0)
    if st == 1:
        score += 1; reasons.append("[Trend] SuperTrend: Yukari")
    elif st == -1:
        score -= 1; reasons.append("[Trend] SuperTrend: Asagi")

    # Haftalik trend konfirmasyonu
    wt = result.get("weekly_trend")
    if wt == "Yukari":
        score += 1; reasons.append("[Trend] Haftalik trend: Yukari")
    elif wt == "Asagi":
        score -= 1; reasons.append("[Trend] Haftalik trend: Asagi")

    score = max(-4, min(4, score))
    vote = "AL" if score >= 2 else ("SAT" if score <= -2 else "BEKLE")
    return score, vote, reasons


def _module_momentum(result):
    """Momentum modulu: hisse gucluyor ve piyasayi dovor mu?"""
    score, reasons = 0, []

    # RS Rating — BIST100'e gore goreceli guc
    rs = result.get("rs_rating")
    if rs is not None:
        if rs >= 5:
            score += 1; reasons.append(f"[Momentum] RS: BIST100 uzerinde (+{rs:.1f}%)")
        elif rs <= -10:
            score -= 1; reasons.append(f"[Momentum] RS: BIST100 gerisinde ({rs:.1f}%)")

    # MACD — momentum yonu
    macd, macd_sig = result.get("macd"), result.get("macd_signal")
    if macd is not None and macd_sig is not None:
        if macd > macd_sig and macd > 0:
            score += 1; reasons.append("[Momentum] MACD: Pozitif ve yukselme")
        elif macd < macd_sig and macd < 0:
            score -= 1; reasons.append("[Momentum] MACD: Negatif ve dusus")

    # EMA crossover
    ec = result.get("ema_cross", "")
    if ec in ("Golden Cross", "Yukari"):
        score += 1; reasons.append(f"[Momentum] EMA: {ec}")
    elif ec in ("Death Cross", "Asagi"):
        score -= 1; reasons.append(f"[Momentum] EMA: {ec}")

    # Divergence
    if result.get("bull_div", 0):
        score += 1; reasons.append("[Momentum] Boga divergence tespit edildi")
    if result.get("bear_div", 0):
        score -= 1; reasons.append("[Momentum] Ayi divergence tespit edildi")

    score = max(-4, min(4, score))
    vote = "AL" if score >= 2 else ("SAT" if score <= -2 else "BEKLE")
    return score, vote, reasons


def _module_timing(result):
    """Zamanlama modulu: simdi girmek dogru zaman mi?"""
    score, reasons = 0, []

    # RSI — asiri alim/satim filtresi (en kritik)
    rsi = result.get("rsi")
    if rsi is not None:
        if rsi < 30:
            score += 2; reasons.append(f"[Timing] RSI asiri satimda — potansiyel dip ({rsi})")
        elif rsi < 55:
            score += 1; reasons.append(f"[Timing] RSI iyi giris bolgesi ({rsi})")
        elif rsi > 70:
            score -= 2; reasons.append(f"[Timing] RSI asiri alimda — tehlikeli giris ({rsi})")
        elif rsi > 60:
            score -= 1; reasons.append(f"[Timing] RSI yukselmis bolgede ({rsi})")

    # Bollinger bantlari
    bb_u = result.get("bb_upper")
    bb_l = result.get("bb_lower")
    price = result.get("price")
    if bb_u and bb_l and price:
        if price <= bb_l * 1.01:
            score += 1; reasons.append("[Timing] Bollinger alt bandi — ucuz bolge")
        elif price >= bb_u:
            score -= 1; reasons.append("[Timing] Bollinger ust bandi asildi — pahali bolge")

    # Hacim onayi
    vol = result.get("volume_ratio")
    if vol is not None:
        if vol >= 1.0:
            score += 1; reasons.append(f"[Timing] Hacim onayliyor ({vol}x)")
        elif vol < 0.6:
            score -= 1; reasons.append(f"[Timing] Dusuk hacim — guvenilmez ({vol}x)")

    score = max(-4, min(4, score))
    # Timing: tek AL yeterli (zamanlama firsatci olmali)
    vote = "AL" if score >= 1 else ("SAT" if score <= -2 else "BEKLE")
    return score, vote, reasons


def _module_external(result, is_backtest):
    """Harici sinyal modulu: haber, Minervini, sektor, RL."""
    score, reasons = 0, []

    # Haber sentimenti
    ns = result.get("news_sentiment", 0)
    if ns >= 15:
        score += 1; reasons.append(f"[External] Haber: Olumlu ({ns})")
    elif ns <= -15:
        score -= 1; reasons.append(f"[External] Haber: Olumsuz ({ns})")

    if not is_backtest:
        # Minervini trend sablonu
        mv = result.get("minervini_score", 0)
        if mv >= 6:
            score += 1; reasons.append(f"[External] Minervini: Guclu ({mv}/8)")

        # Sektor makro skoru
        ss = result.get("sector_score", 0)
        if ss >= 3:
            score += 1; reasons.append(f"[External] Sektor: Pozitif makro ({ss})")
        elif ss <= -3:
            score -= 1; reasons.append(f"[External] Sektor: Negatif makro ({ss})")

        # RL ajan sinyali
        rl_sig  = result.get("rl_signal", "BEKLE")
        rl_conf = result.get("rl_confidence", 0.0)
        if rl_sig == "AL" and rl_conf >= 0.6:
            score += 1; reasons.append(f"[External] RL Ajan: AL (%{rl_conf*100:.0f} guven)")
        elif rl_sig == "SAT" and rl_conf >= 0.6:
            score -= 1; reasons.append(f"[External] RL Ajan: SAT (%{rl_conf*100:.0f} guven)")

    score = max(-4, min(4, score))
    vote = "AL" if score >= 1 else ("SAT" if score <= -1 else "BEKLE")
    return score, vote, reasons


def analyze_single_stock(symbol, period="1y", interval="1d", profile_name=None, is_index=False, sector_index_score=None, turtle_active=False, df=None, is_backtest=False):
    """
    Tek bir hisse icin analiz. Profil secilirse o profilin
    RSI esikleri ve agirlik katsayilari kullanilir.
    """
    # Profil ayarlarini al
    if profile_name and profile_name in PROFILES:
        profile = PROFILES[profile_name]
    else:
        profile = PROFILES["Manuel"]

    result = {
        "symbol": symbol,
        "signal": "BEKLE",
        "score": 0,
        "reasons": [],
        "price": None,
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "sma_9": None,
        "sma_21": None,
        "sma_50": None,
        "sma_200": None,
        "bb_upper": None,
        "bb_lower": None,
        "bull_div": 0,
        "bear_div": 0,
        "div_details": "",
        "volume_ratio": None,
        "volume_trend": "",
        "adx": None,
        "trend_direction": "",
        "atr": None,
        "atr_stop_loss": None,
        "nearest_support": None,
        "nearest_resistance": None,
        "support_distance_pct": None,
        "resistance_distance_pct": None,
        "weekly_trend": None,
        "ema_9": None,
        "ema_21": None,
        "ema_cross": None,
        "fib_zone": None,
        "fib_382": None,
        "fib_618": None,
        "dc_upper_20": None,
        "dc_lower_20": None,
        "dc_mid_20": None,
        "cci": None,
        "willr": None,
        "anomaly_status": False,
        "anomaly_score": 0,
        "minervini_score": 0,
        "glv_near": False,
        "glv_diff": 0,
        "glv_val": 0,
        "analyst_reco": "Nötr",
        "analyst_target": 0,
        "analyst_count": 0,
        "pe_ratio": None,
        "pb_ratio": None,
        "dividend_yield": None,
        "market_cap": None,
        "sector": None,
        "cluster_name": "Bilinmiyor",
        "debt_to_equity": None,
        "roe": None,
        "revenue_growth": None,
        "free_cashflow": None,
        "week52_position": None,
        "sector_score": 0,
        "macro_info": {},
        "rwb_highway": False,
        "rwb_score": 0,
        "sim_expected_price": 0,
        "sim_success_prob": 0,
        "sim_risk_pct": 0,
        "champion_indicator": None,
        "news_sentiment": 0,
        "news_items": [],
        "catalysts": {},
        "seasonal_stats": [],
        "current_month": 0,
        "supertrend_dir": 0,
        "z_score": 0,
        "gmma_div": 0,
        "gmma_spread": 0,
        "atr_trailing_stop": 0,
        "volume_peak_risk": 0,
        "rs_rating": None,       # Göreceli Güç: hissenin BIST100'e göre fazla/eksi getirisi (%)
        "module_votes":  {"trend": "BEKLE", "momentum": "BEKLE", "timing": "BEKLE", "external": "BEKLE"},
        "module_scores": {"trend": 0, "momentum": 0, "timing": 0, "external": 0},
        "rl_signal": "BEKLE",
        "rl_confidence": 0.0,
        "profile": profile_name or "Manuel",
        "df": None,
    }
    
    # 1. Veri cek (Eğer dışarıdan df gelmediyse)
    if df is None:
        df = fetch_stock_data(symbol, period=period, interval=interval)
        if df is None or df.empty:
            result["reasons"].append("Veri cekilemedi")
            return result
    
    # 2. Indikatorler (Eğer dışarıdan df geldiyse ve kolonlar doluysa atla)
    if "SMA_9" not in df.columns:
        df = add_technical_indicators(df)
    
    # 3. Divergence (Aynı şekilde eğer varsa atla)
    if "bull_div" not in df.columns:
        df = detect_divergences(df, left_bars=3, right_bars=3)
    div_info = get_recent_divergence(df, lookback=25)
    
    # Son veriler
    # Son veriler
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    result["price"] = round(last["Close"], 2)
    result["rsi"] = round(last["RSI_14"], 2) if last["RSI_14"] == last["RSI_14"] else None
    result["macd"] = round(last["MACD"], 4) if last["MACD"] == last["MACD"] else None
    result["macd_signal"] = round(last["MACD_Signal"], 4) if last["MACD_Signal"] == last["MACD_Signal"] else None
    result["sma_9"] = round(last["SMA_9"], 2) if last["SMA_9"] == last["SMA_9"] else None
    result["sma_21"] = round(last["SMA_21"], 2) if last["SMA_21"] == last["SMA_21"] else None
    result["sma_50"] = round(last["SMA_50"], 2) if last["SMA_50"] == last["SMA_50"] else None
    result["sma_200"] = round(last["SMA_200"], 2) if "SMA_200" in df.columns and last["SMA_200"] == last["SMA_200"] else None
    result["bb_upper"] = round(last["BB_Upper"], 2) if last["BB_Upper"] == last["BB_Upper"] else None
    result["bb_lower"] = round(last["BB_Lower"], 2) if last["BB_Lower"] == last["BB_Lower"] else None
    result["minervini_score"] = last["MINERVINI_SCORE"] if "MINERVINI_SCORE" in df.columns and last["MINERVINI_SCORE"] == last["MINERVINI_SCORE"] else 0
    # --- ANOMALİ TESPİTİ (Isolation Forest) ---
    if not is_backtest:
        try:
            is_anomaly, anomaly_score = anomaly_detector.get_anomaly_status(df)
            result["anomaly_status"] = is_anomaly
            result["anomaly_score"] = anomaly_score
        except Exception as e:
            print(f"[UYARI] Anomali tespiti hatası: {e}")
            result["anomaly_status"] = False
            result["anomaly_score"] = 0
    else:
        result["anomaly_status"] = False
        result["anomaly_score"] = 0
    
    # --- MINERVINI & GREEN LINE (Elite Plus) ---
    result["glv_val"] = round(last["GLV_VAL"], 2) if "GLV_VAL" in df.columns else 0
    result["bear_div"] = div_info["bearish"]
    result["bull_div"] = div_info["bullish"]
    # --- STRATEGY OPTIMIZER (Faz 26) ---
    if not is_backtest:
        champ_ind = optimizer.get_champion(symbol, df)
        if champ_ind:
            result["champion_indicator"] = champ_ind["name"]

    # --- NEWS SENTIMENT (Faz 27) ---
    if not is_backtest:
        news_score, news_list = news_scraper.get_sentiment_score(symbol)
        result["news_sentiment"] = news_score
        result["news_items"] = news_list

    # --- CATALYST TRACKING (Faz 27) ---
    if not is_backtest:
        result["catalysts"] = catalyst_manager.get_market_catalysts(symbol) or {}

    # --- SEASONAL ANALYSIS (Faz 28) ---
    seas_res = seasonal_analyzer.get_seasonal_report(df)
    if seas_res:
        result["seasonal_stats"] = seas_res["stats"]
        result["current_month"] = seas_res["current_month"]
    
    # --- ELITE VISION (Faz 29) ---
    result["supertrend_dir"] = last["SUPERTREND_DIR"] if "SUPERTREND_DIR" in df.columns else 0
    result["z_score"] = last["Z_SCORE"] if "Z_SCORE" in df.columns else 0
    result["gmma_div"] = last["GMMA_DIVERGENCE"] if "GMMA_DIVERGENCE" in df.columns else 0
    result["gmma_spread"] = last["GMMA_SPREAD"] if "GMMA_SPREAD" in df.columns else 0
    result["rwb_highway"] = last["RWB_HIGHWAY"] if "RWB_HIGHWAY" in df.columns else False
    result["rwb_score"] = last["RWB_SCORE"] if "RWB_SCORE" in df.columns else 0

    # --- MONTE CARLO SIMULATION (Faz 25) ---
    sim_res = sim_manager.get_monte_carlo_results(symbol, df)
    if sim_res:
        result["sim_expected_price"] = sim_res["expected_price"]
        result["sim_success_prob"] = sim_res["success_probability"]
        result["sim_risk_pct"] = sim_res["max_risk_pct"]
    
    # ADX
    adx_val = last["ADX"] if "ADX" in df.columns and last["ADX"] == last["ADX"] else None
    plus_di = last["PLUS_DI"] if "PLUS_DI" in df.columns and last["PLUS_DI"] == last["PLUS_DI"] else None
    minus_di = last["MINUS_DI"] if "MINUS_DI" in df.columns and last["MINUS_DI"] == last["MINUS_DI"] else None
    result["adx"] = round(adx_val, 2) if adx_val else None
    
    if adx_val and plus_di and minus_di:
        if plus_di > minus_di:
            result["trend_direction"] = "Yukari"
        else:
            result["trend_direction"] = "Asagi"
    
    # ATR
    atr_val = last["ATR_14"] if "ATR_14" in df.columns and last["ATR_14"] == last["ATR_14"] else None
    if atr_val and result["price"]:
        result["atr"] = round(atr_val, 2)
        result["atr_stop_loss"] = round(result["price"] - (atr_val * 2), 2)
    
    # Destek / Direnc
    if len(df) >= 40:
        sr = find_support_resistance(df, window=20, num_levels=3)
        result["nearest_support"] = sr["nearest_support"]
        result["nearest_resistance"] = sr["nearest_resistance"]
        result["support_distance_pct"] = sr["support_distance_pct"]
        result["resistance_distance_pct"] = sr["resistance_distance_pct"]
    
    # --- GÖRECELI GÜÇ (RS Rating) vs BIST100 ---
    # Hissenin son 6 aylık getirisini BIST100 ile karşılaştırır.
    # Endeks analizi ve backtest'te atlanır.
    if not is_index and not is_backtest:
        try:
            df_bist = fetch_stock_data("XU100.IS", period=period, interval=interval)
            if df_bist is not None and len(df_bist) >= 20:
                lookback = min(126, len(df) - 1, len(df_bist) - 1)
                stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100
                index_ret = (df_bist["Close"].iloc[-1] / df_bist["Close"].iloc[-lookback] - 1) * 100
                result["rs_rating"] = round(stock_ret - index_ret, 2)
        except Exception as e:
            print(f"[RS Rating] {symbol} hesaplama hatası: {e}")

    # Multi-Timeframe: Haftalik trend kontrolu (sadece canli modda)
    weekly_trend = None
    if not is_backtest and interval in ["1d", "15m", "1h"]:
        try:
            df_weekly = fetch_stock_data(symbol, period="2y", interval="1wk")
            if df_weekly is not None and len(df_weekly) > 30:
                df_weekly = add_technical_indicators(df_weekly)
                wk_last = df_weekly.iloc[-1]
                wk_adx = wk_last["ADX"] if wk_last["ADX"] == wk_last["ADX"] else 0
                wk_plus = wk_last["PLUS_DI"] if wk_last["PLUS_DI"] == wk_last["PLUS_DI"] else 0
                wk_minus = wk_last["MINUS_DI"] if wk_last["MINUS_DI"] == wk_last["MINUS_DI"] else 0
                wk_sma50 = wk_last["SMA_50"] if wk_last["SMA_50"] == wk_last["SMA_50"] else None
                wk_price = wk_last["Close"]

                if wk_adx >= 20 and wk_plus > wk_minus:
                    weekly_trend = "Yukari"
                elif wk_adx >= 20 and wk_minus > wk_plus:
                    weekly_trend = "Asagi"
                elif wk_sma50 and wk_price > wk_sma50:
                    weekly_trend = "Yukari"
                elif wk_sma50 and wk_price < wk_sma50:
                    weekly_trend = "Asagi"
                else:
                    weekly_trend = "Yatay"
        except:
            pass
    result["weekly_trend"] = weekly_trend
    
    # EMA
    ema9 = last["EMA_9"] if "EMA_9" in df.columns and last["EMA_9"] == last["EMA_9"] else None
    ema21 = last["EMA_21"] if "EMA_21" in df.columns and last["EMA_21"] == last["EMA_21"] else None
    prev_ema9 = prev["EMA_9"] if "EMA_9" in df.columns and prev["EMA_9"] == prev["EMA_9"] else None
    prev_ema21 = prev["EMA_21"] if "EMA_21" in df.columns and prev["EMA_21"] == prev["EMA_21"] else None
    
    result["ema_9"] = round(ema9, 2) if ema9 else None
    result["ema_21"] = round(ema21, 2) if ema21 else None
    
    if ema9 and ema21 and prev_ema9 and prev_ema21:
        if prev_ema9 < prev_ema21 and ema9 > ema21:
            result["ema_cross"] = "Golden Cross"
        elif prev_ema9 > prev_ema21 and ema9 < ema21:
            result["ema_cross"] = "Death Cross"
        elif ema9 > ema21:
            result["ema_cross"] = "Yukari"
        else:
            result["ema_cross"] = "Asagi"
    
    # Fibonacci
    fib = None
    if len(df) >= 50:
        fib = calculate_fibonacci(df, lookback=50)
        if fib:
            result["fib_zone"] = fib["zone"]
            result["fib_382"] = fib["fib_382"]
            result["fib_618"] = fib["fib_618"]
            
    # Temel Analiz verisi (Endeksler, backtest ve Trend Avcisi icin atlanir)
    fundamental = None
    if not is_index and not is_backtest and profile_name != "Trend Avcisi":
        try:
            fundamental = fetch_fundamental_data(symbol)
            if fundamental:
                result["pe_ratio"] = fundamental.get("pe_ratio") if fundamental.get("pe_ratio", 0) > 0 else None
                result["pb_ratio"] = fundamental.get("pb_ratio") if fundamental.get("pb_ratio", 0) > 0 else None
                result["dividend_yield"] = fundamental.get("dividend_yield")
                result["market_cap"] = fundamental.get("market_cap")
                result["sector"] = fundamental.get("sector")
                
                # Analist Verileri
                result["analyst_reco"] = fundamental.get("recommendation_key", "Nötr")
                result["analyst_target"] = fundamental.get("target_mean_price", 0)
                result["analyst_count"] = fundamental.get("number_of_analysts", 0)
                
                # Ek Temel Gostergeler
                result["debt_to_equity"] = fundamental.get("debt_to_equity") if fundamental.get("debt_to_equity", 0) > 0 else None
                result["roe"] = fundamental.get("roe") if fundamental.get("roe", 0) != 0 else None
                result["revenue_growth"] = fundamental.get("revenue_growth") if fundamental.get("revenue_growth", 0) != 0 else None
                result["free_cashflow"] = fundamental.get("free_cashflow")
                # 52-hafta pozisyonu
                w52_low = fundamental["fifty_two_week_low"]
                if w52_high > 0 and w52_low > 0 and result["price"]:
                    w52_range = w52_high - w52_low
                    if w52_range > 0:
                        result["week52_position"] = round((result["price"] - w52_low) / w52_range * 100, 1)
        except:
            pass
    
    # Makro veri ve sektor skoru (Endeksler ve backtest icin atlanir)
    macro_data = None
    sector_name = SECTOR_MAP.get(symbol)
    if not is_index and not is_backtest:
        try:
            macro_data = fetch_macro_data()
            result["macro_info"] = macro_data
            
            if sector_name and macro_data and sector_name in SECTOR_MACRO_CORRELATION:
                correlations = SECTOR_MACRO_CORRELATION[sector_name]
                sector_score = 0
                for macro_name, corr_weight in correlations.items():
                    if macro_name in macro_data:
                        m = macro_data[macro_name]
                        # Aylik degisim yuzdesini skorla carp
                        if m["trend"] == "Yukari":
                            sector_score += corr_weight * 2
                        elif m["trend"] == "Asagi":
                            sector_score -= corr_weight * 2
                result["sector_score"] = round(sector_score, 1)
        except:
            pass
    
    # ============================================================
    # HACİM ANALİZİ
    # ============================================================
    vol_avg_20 = df["Volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else None
    current_vol = last["Volume"]
    
    if vol_avg_20 is not None and vol_avg_20 > 0:
        vol_ratio = current_vol / vol_avg_20
        result["volume_ratio"] = round(vol_ratio, 2)
        price_change = last["Close"] - prev["Close"]
        
        if vol_ratio >= 1.5:
            result["volume_trend"] = "Guclu Alis Hacmi" if price_change > 0 else "Guclu Satis Hacmi"
        elif vol_ratio >= 1.0:
            result["volume_trend"] = "Normal Hacim"
        else:
            result["volume_trend"] = "Dusuk Hacim"
    
    # ============================================================
    # MODÜLER SKOR HESABI — 4 Modül Konsensüs Sistemi
    # ============================================================

    # --- 4 Modül cagir ---
    trend_sc,    trend_v,    trend_r    = _module_trend(result)
    momentum_sc, momentum_v, momentum_r = _module_momentum(result)
    timing_sc,   timing_v,   timing_r   = _module_timing(result)
    external_sc, external_v, external_r = _module_external(result, is_backtest)

    # Anomali varsa AL oylarini baskila
    votes = [trend_v, momentum_v, timing_v, external_v]
    if result.get("anomaly_status"):
        votes = ["SAT" if v == "AL" else v for v in votes]
        result["reasons"].append(f"Anomali tespit edildi — AL oylari baskilandi (Skor: {result['anomaly_score']})")

    al_count  = votes.count("AL")
    sat_count = votes.count("SAT")

    # Gerekceler
    result["reasons"].extend(trend_r)
    result["reasons"].extend(momentum_r)
    result["reasons"].extend(timing_r)
    result["reasons"].extend(external_r)

    # Modul detaylari (UI icin)
    result["module_votes"]  = {
        "trend": trend_v, "momentum": momentum_v,
        "timing": timing_v, "external": external_v,
    }
    result["module_scores"] = {
        "trend": trend_sc, "momentum": momentum_sc,
        "timing": timing_sc, "external": external_sc,
    }

    # Profil agirlikli normalize skor (0-100)
    _MW = {
        "Trend Avcisi":      {"trend": 1.5, "momentum": 1.5, "timing": 1.0, "external": 0.7},
        "Deger Yatirimcisi": {"trend": 0.8, "momentum": 0.8, "timing": 1.2, "external": 1.5},
        "Manuel":            {"trend": 1.0, "momentum": 1.0, "timing": 1.0, "external": 1.0},
    }
    mw = _MW.get(profile_name, _MW["Manuel"])
    MODULE_MAX = 4
    weighted_sum = (trend_sc    * mw["trend"]    +
                    momentum_sc * mw["momentum"]  +
                    timing_sc   * mw["timing"]    +
                    external_sc * mw["external"])
    max_sum = MODULE_MAX * (mw["trend"] + mw["momentum"] + mw["timing"] + mw["external"])
    score = 50 + 50 * (weighted_sum / max_sum) if max_sum > 0 else 50

    # ============================================================
    # SKOR SINIRLANDIRMA VE SİNYAL (oy bazli)
    # ============================================================
    score = max(0, min(100, round(score)))
    result["score"] = score

    if al_count >= 3:
        result["signal"] = "GUCLU AL"
    elif al_count == 2:
        result["signal"] = "AL"
    elif al_count == 1:
        result["signal"] = "INCELE"
    elif sat_count >= 3:
        result["signal"] = "SAT"
    elif sat_count == 2:
        result["signal"] = "DIKKAT"
    else:
        result["signal"] = "BEKLE"

    # ── HARD GATES ──────────────────────────────────────────────
    if result["signal"] in ("AL", "GUCLU AL"):
        rsi_ok    = result["rsi"] is None or result["rsi"] < 68
        adx_ok    = result["adx"] is None or result["adx"] >= 20
        volume_ok = result["volume_ratio"] is None or result["volume_ratio"] >= 0.7
        # Yapisal trend gate: fiyat SMA200'un %5+ altindaysa AL engelle
        # (SMA200 yoksa SMA50'ye bak; ikisi de yoksa engelleme)
        _sma200 = result.get("sma_200")
        _sma50  = result.get("sma_50")
        _price  = result.get("price")
        _trend_ma = _sma200 if _sma200 else _sma50
        if _trend_ma and _price:
            struct_ok = _price >= _trend_ma * 0.95
        else:
            struct_ok = True
        if not (rsi_ok and adx_ok and volume_ok and struct_ok):
            result["signal"] = "BEKLE"
            failed = []
            if not rsi_ok:    failed.append(f"RSI asiri alim ({result['rsi']})")
            if not adx_ok:    failed.append(f"ADX zayif ({result['adx']})")
            if not volume_ok: failed.append(f"Dusuk hacim ({result['volume_ratio']}x)")
            if not struct_ok:
                _lbl = "SMA200" if _sma200 else "SMA50"
                failed.append(f"Yapisal dusus (Fiyat {_price:.2f} < {_lbl} {_trend_ma:.2f} * 0.95)")
            result["reasons"].append(f"[Hard Gate] AL engellendi: {', '.join(failed)}")
    # ────────────────────────────────────────────────────────────

    # --- Phase 32: ELITE ULTRA LIVE EXIT ADVISOR ---
    # 1. Dynamic ATR Trailing Stop (Live)
    if result["price"] and result["atr"]:
        # ATR çarpanı backtestteki gibi 2.5
        atr_dist = result["atr"] * 2.5
        result["atr_trailing_stop"] = round(result["price"] - atr_dist, 2)
        
    # 2. Volume Blow-off Top Risk Scoring
    if len(df) >= 20:
        last_20 = df.iloc[-20:]
        avg_vol_20 = last_20["Volume"].mean()
        max_price_20 = last_20["Close"].max()
        current_vol = df.iloc[-1]["Volume"]
        current_price = df.iloc[-1]["Close"]
        
        v_risk = 0
        # Fiyat zirveye ne kadar yakın? (Son 20 günün %2'lik bandı içindeyse zirve kabul edilir)
        is_near_peak = current_price >= (max_price_20 * 0.98)
        # Hacim ortalamanın ne kadar üzerinde?
        vol_ratio_20 = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        
        if is_near_peak:
            if vol_ratio_20 > 2.5: v_risk = 90 # Tehlike!
            elif vol_ratio_20 > 1.8: v_risk = 65 # Uyarı
            elif vol_ratio_20 > 1.3: v_risk = 40 # Dikkat
            
        result["volume_peak_risk"] = v_risk
        if v_risk >= 65:
            result["reasons"].append(f"🔴 Zirve Riski: Hacim son 20 gün ortalamasının {vol_ratio_20:.1f} katına çıktı! (Blow-off Top Riski)")
    
    if not result["reasons"]:
        result["reasons"].append("Belirgin bir sinyal yok")
    
    result["df"] = df
    return result

def scan_all_bist50(period="6mo", interval="1d", profile_name=None):
    """Tum BIST50 hisselerini tarar."""
    from config import BIST50_SYMBOLS
    
    results = []
    for i, symbol in enumerate(BIST50_SYMBOLS):
        print(f"  [{i+1}/{len(BIST50_SYMBOLS)}] {symbol} taraniyor...")
        result = analyze_single_stock(symbol, period=period, interval=interval, profile_name=profile_name)
        results.append(result)
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    print("=" * 55)
    print("  PROFİL BAZLI STRATEJİ TESTİ")
    print("=" * 55)
    
    test_symbol = "THYAO.IS"
    
    for pname in ["Trend Avcisi", "Deger Yatirimcisi"]:
        p = PROFILES[pname]
        print(f"\n--- [{pname}] {test_symbol} ({p['period']}/{p['interval']}) ---")
        result = analyze_single_stock(test_symbol, period=p["period"], interval=p["interval"], profile_name=pname)
        
        print(f"  Fiyat:  {result['price']} TL")
        print(f"  RSI:    {result['rsi']}")
        print(f"  Hacim:  {result['volume_ratio']}x ({result['volume_trend']})")
        print(f"  Div:    Bull:{result['bull_div']} Bear:{result['bear_div']}")
        print(f"  Sinyal: {result['signal']} (Skor: {result['score']})")
        for r in result["reasons"]:
            print(f"    - {r}")
