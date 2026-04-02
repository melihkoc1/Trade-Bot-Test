"""
Combo Backtester — Rotasyon + RL + Temel/Teknik Filtreler

Çalışma mantığı (her bar):
  1. Rotasyon momentum skoru ile en iyi hisseyi seç (entry_threshold)
  2. O hisse için filtreler uygula:
       a. MA200: fiyat SMA200 üstünde mi?
       b. F-Score: Piotroski ≥ min_fscore?
       c. Sektör endeks filtresi: XBANK/XELKT/... MA50 üstünde mi?
       d. RL çıkış sinyali: pozisyon içinde SAT diyorsa çık
  3. Tüm filtreler geçerse pozisyona gir
  4. Çıkış: rotasyon trailing stop, skor düşüşü veya RL SAT sinyali
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

from indicators import (
    add_technical_indicators, calculate_sma,
    calculate_atr, calculate_supertrend, detect_divergences,
)
from rl_environment import _compute_features, N_FEATURES
from config import SECTOR_MAP, SECTOR_INDEX_MAP

COMMISSION = 0.001
MODEL_PATH = "models/ppo_tradebot.zip"
BEST_PATH  = "models/best_model.zip"


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _fetch(symbol: str, period: str = None,
           start: str = None, end: str = None) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(symbol)
        if start and end:
            df = t.history(start=start, end=end, interval="1d")
        else:
            df = t.history(period=period or "2y", interval="1d")
        if df is None or df.empty or len(df) < 250:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["Volume"] = df["Volume"].replace(0, 1)
        return df
    except Exception:
        return None


def _load_rl_model():
    """En iyi RL modelini yükle. Yoksa None döndür."""
    try:
        from stable_baselines3 import PPO
        path = BEST_PATH if os.path.exists(BEST_PATH) else MODEL_PATH
        if not os.path.exists(path):
            return None
        return PPO.load(path)
    except Exception:
        return None


def _rotation_score(df: pd.DataFrame, i: int) -> float:
    """Rotasyon momentum skoru (0-100) — rotator.py ile aynı mantık."""
    row  = df.iloc[i]
    sc   = 50.0
    rsi  = row.get("RSI_14", 50)
    if 45 <= rsi <= 65:   sc += 8
    elif rsi > 65:        sc -= 5
    elif rsi < 35:        sc -= 10

    if row.get("MACD_Hist", 0) > 0:  sc += 8
    else:                              sc -= 5

    sma21 = row.get("SMA_21", 0); close = row.get("Close", 0)
    if sma21 > 0: sc += 10 if close > sma21 else -8
    sma50 = row.get("SMA_50", 0)
    if sma50 > 0: sc += 8  if close > sma50 else -6

    # ADX — indicators.py'deki gerçek kolon adı "ADX"
    adx = row.get("ADX", 0)
    if adx > 30:   sc += 10
    elif adx > 20: sc += 5

    vol    = row.get("Volume", 0)
    vol_ma = df["Volume"].iloc[max(0, i-20):i].mean() if i > 0 else vol
    if vol_ma > 0 and vol > vol_ma * 1.3: sc += 5

    # Bollinger — indicators.py'deki gerçek kolon adları "BB_Upper" / "BB_Lower"
    bb_u = row.get("BB_Upper", 0); bb_l = row.get("BB_Lower", 0)
    if bb_u > 0 and bb_l > 0:
        if close > bb_u:                      sc -= 8
        elif close > (bb_u + bb_l) / 2:      sc += 5

    # SuperTrend yönü — pozitif trend güçlendirici
    st_dir = row.get("SUPERTREND_DIR", 0)
    if st_dir == 1:    sc += 7
    elif st_dir == -1: sc -= 7

    return float(np.clip(sc, 0, 100))


def _rl_obs_exit(feat_df: pd.DataFrame, idx: int, position: dict, cur_price: float) -> np.ndarray:
    """
    Pozisyon içindeyken RL için stateful gözlem üret (çıkış kontrolü).
    Model bu şekilde eğitildi — in_position=1 + gerçek PnL/süre/drawdown.
    """
    row = feat_df.iloc[idx].values.astype(np.float32)
    unreal_pnl = np.clip(
        (cur_price - position["buy_price"]) / position["buy_price"], -0.5, 0.5
    )
    days_norm = min(position["days_held"] / 60.0, 1.0)
    pos_dd = np.clip(
        max(0.0, (position["peak_price"] - cur_price) / max(position["peak_price"], 1e-9)),
        0.0, 0.3,
    ) / 0.3
    pos_features = np.array(
        [1.0, unreal_pnl, days_norm, 0.0, pos_dd], dtype=np.float32
    )
    return np.concatenate([row, pos_features])


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def run_combo_backtest(
    symbols: list,
    period: str = "2y",
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100_000.0,
    entry_threshold: float = 62.0,
    trail_pct: float = 8.0,
    min_hold_days: int = 5,
    exit_score_threshold: float = 45.0,
    use_ma200: bool = True,
    min_fscore: int = 5,
    use_rl: bool = True,
    use_sector_filter: bool = False,
    progress_callback=None,
) -> dict:
    """
    Rotasyon + RL + MA200 + F-Score + Sektör Endeksi birleşik backtest.
    """

    total_steps = len(symbols) + 4
    step = 0

    def _prog(msg):
        nonlocal step
        if progress_callback:
            progress_callback(min(step / total_steps, 0.99), msg)
        step += 1

    # ── 1. RL modeli yükle ───────────────────────────────────────────────
    rl_model = None
    if use_rl:
        _prog("RL modeli yükleniyor...")
        rl_model = _load_rl_model()
        if rl_model is None:
            use_rl = False

    # ── 2. F-Score önbellekle ────────────────────────────────────────────
    fscore_map: dict = {}
    if min_fscore > 0:
        _prog("F-Score verileri çekiliyor...")
        try:
            from fundamental import get_fscore_filter
            for sym in symbols:
                fscore_map[sym] = get_fscore_filter(sym)
        except Exception:
            pass

    # ── 3. Sektör endeksi Close + MA50 serileri ─────────────────────────
    # sector_close: {idx_sym: pd.Series}  (tz-naive tarih → kapanış fiyatı)
    # sector_ma50:  {idx_sym: pd.Series}  (tz-naive tarih → MA50)
    sector_close: dict = {}
    sector_ma50:  dict = {}
    if use_sector_filter:
        _prog("Sektör endeksleri çekiliyor...")
        unique_indices = set(SECTOR_INDEX_MAP.values())
        for idx_sym in unique_indices:
            try:
                if start_date and end_date:
                    idx_df = yf.Ticker(idx_sym).history(start=start_date, end=end_date, interval="1d")
                else:
                    idx_df = yf.Ticker(idx_sym).history(period=period or "2y", interval="1d")
                if idx_df.empty or len(idx_df) < 60:
                    continue
                if idx_df.index.tz is not None:
                    idx_df.index = idx_df.index.tz_convert(None)
                idx_df.index = idx_df.index.normalize()
                sector_close[idx_sym] = idx_df["Close"]
                sector_ma50[idx_sym]  = idx_df["Close"].rolling(50).mean()
            except Exception:
                pass

    # ── 4. Veri çek & indikatör + RL feature ────────────────────────────
    _prog("Veri çekiliyor...")
    stock_data:    dict = {}
    rl_features:   dict = {}
    sma200_arrays: dict = {}

    # BIST100 getirisi (RL özellik hesabı için)
    try:
        bist_df   = yf.Ticker("XU100.IS").history(period=period, interval="1d")
        bist_ret5 = bist_df["Close"].pct_change(5) if not bist_df.empty else None
    except Exception:
        bist_ret5 = None

    for sym in symbols:
        _prog(f"Hazırlanıyor: {sym.replace('.IS','')}")
        df = _fetch(sym, period=period, start=start_date, end=end_date)
        if df is None:
            continue
        df = add_technical_indicators(df)
        df = detect_divergences(df, left_bars=5, right_bars=5)
        stock_data[sym] = df

        # SMA200 dizisi (MA200 filtresi için)
        if use_ma200:
            sma200_arrays[sym] = calculate_sma(df["Close"], 200).values

        # RL feature matrisi
        if use_rl and rl_model is not None:
            try:
                rl_features[sym] = _compute_features(df, bist_ret5)
            except Exception:
                pass

    if not stock_data:
        return {"error": "Yeterli veri bulunamadı."}

    # ── 4. Ortak tarih ekseni ────────────────────────────────────────────
    all_dates = None
    for df in stock_data.values():
        idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
        s   = set(idx.normalize())
        all_dates = s if all_dates is None else all_dates & s
    all_dates = sorted(all_dates)

    # stock_data index'lerini tz-naive yap (rotator.py ile aynı)
    for sym in stock_data:
        idx = stock_data[sym].index
        if idx.tz is not None:
            stock_data[sym].index = idx.tz_convert(None)

    if len(all_dates) < 20:
        return {"error": "Ortak tarih yeterli değil."}

    # ── 5. Simülasyon döngüsü ────────────────────────────────────────────
    capital         = initial_capital
    position        = None
    trades          = []
    portfolio_curve = {}
    daily_positions = []
    filter_log      = []   # hangi filtreler tetiklendi?
    warmup          = 30

    for bar_idx, date in enumerate(all_dates):
        # Portföy değeri
        if position is not None:
            sym_df   = stock_data[position["sym"]]
            day_row  = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
            cur_price = float(day_row.iloc[-1]["Close"]) if not day_row.empty else position["buy_price"]
            pv = capital + position["shares"] * cur_price
        else:
            pv = capital

        portfolio_curve[date] = pv
        daily_positions.append({
            "Tarih":   date,
            "Hisse":   position["sym"].replace(".IS", "") if position else "NAKİT",
            "Portföy": round(pv, 2),
        })

        if bar_idx < warmup:
            continue

        # ── Çıkış kontrolü ──────────────────────────────────────────────
        if position is not None:
            sym    = position["sym"]
            sym_df = stock_data[sym]
            day_rows = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
            if day_rows.empty:
                continue
            row = day_rows.iloc[-1]
            cur_price = float(row["Close"])
            position["peak_price"] = max(position["peak_price"], cur_price)
            position["days_held"] += 1

            trail_stop = position["peak_price"] * (1 - trail_pct / 100)
            sym_i = sym_df.index.normalize().get_loc(pd.Timestamp(date).normalize())
            if hasattr(sym_i, "__iter__"):
                sym_i = list(sym_i)[-1]
            cur_score = _rotation_score(sym_df, sym_i) if sym_i >= warmup else 50

            exit_reason = None
            if cur_price <= trail_stop and position["days_held"] >= min_hold_days:
                exit_reason = f"Trailing Stop (%{trail_pct})"
            elif cur_score < exit_score_threshold and position["days_held"] >= min_hold_days:
                exit_reason = f"Skor Düştü ({cur_score:.0f})"
            elif (use_rl and rl_model is not None
                  and sym in rl_features
                  and position["days_held"] >= min_hold_days):
                feat_df = rl_features[sym]
                if sym_i < len(feat_df):
                    obs = _rl_obs_exit(feat_df, sym_i, position, cur_price)
                    rl_action, _ = rl_model.predict(obs, deterministic=True)
                    if int(rl_action) == 2:
                        exit_reason = "RL SAT Sinyali"

            if exit_reason:
                revenue = position["shares"] * cur_price * (1 - COMMISSION)
                profit  = revenue - position["shares"] * position["buy_price"]
                capital += revenue
                trades.append({
                    "Hisse":        sym.replace(".IS", ""),
                    "Giriş Tarihi": position["buy_date"].strftime("%Y-%m-%d"),
                    "Giriş Fiyatı": round(position["buy_price"], 2),
                    "Çıkış Tarihi": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "Çıkış Fiyatı": round(cur_price, 2),
                    "Kâr/Zarar":    round(profit, 2),
                    "Getiri %":     round((cur_price / position["buy_price"] - 1) * 100, 2),
                    "Çıkış Nedeni": exit_reason,
                    "Süre (gün)":   position["days_held"],
                })
                position = None

        # ── Giriş: en yüksek skorlu + filtrelerden geçen hisseyi bul ────
        if position is None:
            candidates = []

            for sym, sym_df in stock_data.items():
                day_rows = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
                if day_rows.empty:
                    continue
                sym_i = sym_df.index.normalize().get_loc(pd.Timestamp(date).normalize())
                if hasattr(sym_i, "__iter__"):
                    sym_i = list(sym_i)[-1]
                if sym_i < warmup:
                    continue

                sc = _rotation_score(sym_df, sym_i)
                if sc <= entry_threshold:
                    continue

                candidates.append((sc, sym, sym_df, sym_i))

            # Skor yüksekten düşüğe sırala — ilk geçeni al
            candidates.sort(key=lambda x: x[0], reverse=True)

            for sc, sym, sym_df, sym_i in candidates:
                cur_price = float(sym_df.iloc[sym_i]["Close"])
                blocked_by = []

                # — Filtre A: MA200 ─────────────────────────────────────
                if use_ma200 and sym in sma200_arrays:
                    ma200_val = sma200_arrays[sym][sym_i]
                    if not np.isnan(ma200_val) and cur_price < ma200_val:
                        blocked_by.append("MA200")
                        continue

                # — Filtre B: F-Score ───────────────────────────────────
                if min_fscore > 0 and sym in fscore_map:
                    fs = fscore_map[sym]
                    if fs is not None and fs < min_fscore:
                        blocked_by.append(f"F-Score({fs})")
                        continue

                # — Filtre C: Sektör Endeksi ────────────────────────────
                # Hissenin sektör endeksi kendi MA50'sinin altındaysa girme
                if use_sector_filter:
                    sector_name = SECTOR_MAP.get(sym)
                    idx_sym = SECTOR_INDEX_MAP.get(sector_name) if sector_name else None
                    if idx_sym and idx_sym in sector_close:
                        ts = pd.Timestamp(date).normalize()
                        cl  = sector_close[idx_sym]
                        ma  = sector_ma50[idx_sym]
                        cl_slice  = cl[cl.index <= ts]
                        ma_slice  = ma[ma.index <= ts]
                        if not cl_slice.empty and not ma_slice.empty:
                            idx_price = float(cl_slice.iloc[-1])
                            ma50_val  = float(ma_slice.iloc[-1])
                            if not np.isnan(ma50_val) and idx_price < ma50_val:
                                blocked_by.append(f"Sektör({sector_name}↓)")
                                continue

                # Tüm filtreler geçti → giriş yap
                # Not: RL çıkış sinyali olarak kullanılıyor (giriş için değil)
                shares = int(capital * 0.95 / (cur_price * (1 + COMMISSION)))
                if shares > 0:
                    cost = shares * cur_price * (1 + COMMISSION)
                    capital -= cost
                    position = {
                        "sym":        sym,
                        "shares":     shares,
                        "buy_price":  cur_price,
                        "buy_date":   pd.Timestamp(date),
                        "peak_price": cur_price,
                        "days_held":  0,
                        "score_in":   round(sc, 1),
                    }
                    filter_log.append({
                        "Tarih":  pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "Hisse":  sym.replace(".IS", ""),
                        "Skor":   round(sc, 1),
                        "Filtre": "✅ Geçti",
                    })
                break  # ilk geçen adayda dur
            else:
                # Tüm adaylar filtreden döndü
                if candidates:
                    filter_log.append({
                        "Tarih":  pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "Hisse":  "—",
                        "Skor":   round(candidates[0][0], 1),
                        "Filtre": f"❌ Bloke ({', '.join(blocked_by)})" if blocked_by else "❌ Bloke",
                    })

    # ── Açık pozisyonu kapat ─────────────────────────────────────────────
    if position is not None:
        last_date = all_dates[-1]
        sym_df    = stock_data[position["sym"]]
        day_rows  = sym_df[sym_df.index.normalize() == pd.Timestamp(last_date).normalize()]
        if not day_rows.empty:
            cur_price = float(day_rows.iloc[-1]["Close"])
            revenue   = position["shares"] * cur_price * (1 - COMMISSION)
            profit    = revenue - position["shares"] * position["buy_price"]
            capital  += revenue
            trades.append({
                "Hisse":        position["sym"].replace(".IS", ""),
                "Giriş Tarihi": position["buy_date"].strftime("%Y-%m-%d"),
                "Giriş Fiyatı": round(position["buy_price"], 2),
                "Çıkış Tarihi": pd.Timestamp(last_date).strftime("%Y-%m-%d"),
                "Çıkış Fiyatı": round(cur_price, 2),
                "Kâr/Zarar":    round(profit, 2),
                "Getiri %":     round((cur_price / position["buy_price"] - 1) * 100, 2),
                "Çıkış Nedeni": "Dönem Sonu",
                "Süre (gün)":   position["days_held"],
            })

    # ── Metrikler ────────────────────────────────────────────────────────
    curve        = pd.Series(portfolio_curve)
    total_return = (capital / initial_capital - 1) * 100
    returns      = curve.pct_change().dropna()
    sharpe       = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    peak         = curve.cummax()
    max_dd       = float(((curve - peak) / peak).min() * 100)
    win_rate     = (sum(1 for t in trades if t["Getiri %"] > 0) / len(trades) * 100) if trades else 0.0

    if progress_callback:
        progress_callback(1.0, "Tamamlandı")

    return {
        "portfolio_curve":  curve,
        "trades":           trades,
        "filter_log":       pd.DataFrame(filter_log) if filter_log else pd.DataFrame(),
        "daily_positions":  pd.DataFrame(daily_positions),
        "filters_used": {
            "ma200":   use_ma200,
            "fscore":  min_fscore,
            "rl":      use_rl and rl_model is not None,
            "sektor":  use_sector_filter and bool(sector_close),
        },
        "metrics": {
            "Toplam Getiri (%)": round(total_return, 2),
            "Sharpe":            round(sharpe, 3),
            "Max Düşüş (%)":     round(max_dd, 2),
            "İşlem Sayısı":      len(trades),
            "Kazanma (%)":       round(win_rate, 1),
            "Final Portföy":     round(capital, 2),
        },
    }
