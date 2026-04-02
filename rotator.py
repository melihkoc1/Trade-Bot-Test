"""
Rotasyon Backtester — Tüm BIST50'yi tarar, en güçlü trende girer,
kâr alıp başka fırsata geçer.

Çalışma mantığı (her bar):
  1. Pozisyon yok → en yüksek skorlu hisseyi al (eşik: entry_threshold)
  2. Pozisyon var → trailing stop veya skor düşüşünde çık, yenisine geç
  3. Nakit bekle

Skor: hızlı teknik momentum skoru (indikatörler önceden hesaplanır)
"""

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import (
    add_technical_indicators, calculate_atr,
    calculate_supertrend, detect_divergences,
)


COMMISSION = 0.001   # %0.1 her işlemde


# ─────────────────────────────────────────────────────────────────────────────
def _fetch(symbol: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df is None or df.empty or len(df) < 60:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["Volume"] = df["Volume"].replace(0, 1)
        return df
    except Exception:
        return None


def _score(df: pd.DataFrame, i: int) -> float:
    """
    Tek bar için hızlı momentum skoru (0-100).
    indikatörler önceden df'e eklenmiş olmalı.
    """
    row = df.iloc[i]
    score = 50.0

    # RSI (30-70 bandı)
    rsi = row.get("RSI_14", 50)
    if 45 <= rsi <= 65:
        score += 8
    elif rsi > 65:
        score -= 5   # aşırı alım
    elif rsi < 35:
        score -= 10

    # MACD histogram pozitif mi?
    macd_h = row.get("MACD_Hist", 0)
    score += 8 if macd_h > 0 else -5

    # Fiyat SMA21 üstünde mi?
    sma21 = row.get("SMA_21", 0)
    close = row.get("Close", 0)
    if sma21 > 0:
        score += 10 if close > sma21 else -8

    # Fiyat SMA50 üstünde mi?
    sma50 = row.get("SMA_50", 0)
    if sma50 > 0:
        score += 8 if close > sma50 else -6

    # ADX trend gücü — gerçek kolon adı "ADX"
    adx = row.get("ADX", 0)
    if adx > 30:
        score += 10
    elif adx > 20:
        score += 5

    # Hacim artışı
    vol     = row.get("Volume", 0)
    vol_ma  = df["Volume"].iloc[max(0, i-20):i].mean() if i > 0 else vol
    if vol_ma > 0 and vol > vol_ma * 1.3:
        score += 5

    # Bollinger — gerçek kolon adları "BB_Upper" / "BB_Lower"
    bb_upper = row.get("BB_Upper", 0)
    bb_lower = row.get("BB_Lower", 0)
    if bb_upper > 0 and bb_lower > 0:
        if close > bb_upper:
            score -= 8   # aşırı uzatılmış
        elif close > (bb_upper + bb_lower) / 2:
            score += 5

    # SuperTrend yönü — pozitif trend güçlendirici
    st_dir = row.get("SUPERTREND_DIR", 0)
    if st_dir == 1:    score += 7
    elif st_dir == -1: score -= 7

    return float(np.clip(score, 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
def run_rotation_backtest(
    symbols: list[str],
    period: str = "1y",
    initial_capital: float = 100_000,
    entry_threshold: float = 62,
    trail_pct: float = 8.0,
    min_hold_days: int = 5,
    exit_score_threshold: float = 45,
    progress_callback=None,
) -> dict:
    """
    Rotasyon backtestini çalıştırır.

    Returns dict:
        portfolio_curve : pd.Series (tarih → portföy değeri)
        trades          : list[dict]
        metrics         : dict
        daily_positions : pd.DataFrame (hangi günde hangi hisse)
    """

    # ── 1. Veri çek & indikatör ekle ───────────────────────────────────
    stock_data: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for idx, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(idx / total, f"Veri çekiliyor: {sym}")
        df = _fetch(sym, period)
        if df is None:
            continue
        df = add_technical_indicators(df)
        df = detect_divergences(df, left_bars=5, right_bars=5)
        stock_data[sym] = df

    if not stock_data:
        return None

    # ── 2. Ortak tarih ekseni ────────────────────────────────────────────
    # Tüm hisselerin kesişim tarihleri — tz-naive olarak sakla
    all_dates = None
    for df in stock_data.values():
        idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
        dates = set(idx.normalize())
        all_dates = dates if all_dates is None else all_dates & dates
    all_dates = sorted(all_dates)
    # stock_data index'lerini de tz-naive yap
    for sym in stock_data:
        idx = stock_data[sym].index
        if idx.tz is not None:
            stock_data[sym].index = idx.tz_convert(None)

    if len(all_dates) < 20:
        return None

    # ── 3. Simülasyon ────────────────────────────────────────────────────
    capital        = initial_capital
    position       = None   # {sym, shares, buy_price, buy_date, peak_price, days_held}
    trades         = []
    portfolio_curve = {}
    daily_positions = []

    warmup = 30   # ilk 30 bar indikatör ısınması

    for bar_idx, date in enumerate(all_dates):
        # Portföy değeri
        if position is not None:
            sym_df = stock_data[position["sym"]]
            day_row = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
            if day_row.empty:
                cur_price = position["buy_price"]
            else:
                cur_price = float(day_row.iloc[-1]["Close"])
            portfolio_val = capital + position["shares"] * cur_price
        else:
            portfolio_val = capital

        portfolio_curve[date] = portfolio_val

        daily_positions.append({
            "Tarih": date,
            "Hisse": position["sym"].replace(".IS", "") if position else "NAKİT",
            "Portföy": round(portfolio_val, 2),
        })

        if bar_idx < warmup:
            continue

        # ── Mevcut pozisyon çıkış kontrolü ──────────────────────────────
        if position is not None:
            sym = position["sym"]
            sym_df = stock_data[sym]
            day_rows = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
            if day_rows.empty:
                continue
            row = day_rows.iloc[-1]
            cur_price = float(row["Close"])

            # Peak güncelle
            position["peak_price"] = max(position["peak_price"], cur_price)
            position["days_held"] += 1

            # Trailing stop
            trail_stop = position["peak_price"] * (1 - trail_pct / 100)

            # Güncel bar index bu hisse için
            sym_i = sym_df.index.normalize().get_loc(pd.Timestamp(date).normalize())
            if hasattr(sym_i, '__iter__'):
                sym_i = list(sym_i)[-1]

            cur_score = _score(sym_df, sym_i) if sym_i >= warmup else 50

            exit_reason = None
            if cur_price <= trail_stop and position["days_held"] >= min_hold_days:
                exit_reason = f"Trailing Stop (%{trail_pct})"
            elif cur_score < exit_score_threshold and position["days_held"] >= min_hold_days:
                exit_reason = f"Skor Düştü ({cur_score:.0f})"

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

        # ── Giriş: en iyi fırsatı bul ───────────────────────────────────
        if position is None:
            best_sym   = None
            best_score = entry_threshold

            for sym, sym_df in stock_data.items():
                day_rows = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
                if day_rows.empty:
                    continue
                sym_i = sym_df.index.normalize().get_loc(pd.Timestamp(date).normalize())
                if hasattr(sym_i, '__iter__'):
                    sym_i = list(sym_i)[-1]
                if sym_i < warmup:
                    continue
                sc = _score(sym_df, sym_i)
                if sc > best_score:
                    best_score = sc
                    best_sym   = sym

            if best_sym is not None:
                sym_df  = stock_data[best_sym]
                day_rows = sym_df[sym_df.index.normalize() == pd.Timestamp(date).normalize()]
                entry_price = float(day_rows.iloc[-1]["Close"])
                shares = int(capital * 0.95 / (entry_price * (1 + COMMISSION)))
                if shares > 0:
                    cost = shares * entry_price * (1 + COMMISSION)
                    capital -= cost
                    position = {
                        "sym":        best_sym,
                        "shares":     shares,
                        "buy_price":  entry_price,
                        "buy_date":   pd.Timestamp(date),
                        "peak_price": entry_price,
                        "days_held":  0,
                        "score_in":   round(best_score, 1),
                    }

    # ── Açık pozisyonu kapat ─────────────────────────────────────────────
    if position is not None:
        last_date = all_dates[-1]
        sym_df = stock_data[position["sym"]]
        day_rows = sym_df[sym_df.index.normalize() == pd.Timestamp(last_date).normalize()]
        if not day_rows.empty:
            cur_price = float(day_rows.iloc[-1]["Close"])
            revenue = position["shares"] * cur_price * (1 - COMMISSION)
            profit  = revenue - position["shares"] * position["buy_price"]
            capital += revenue
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
    curve = pd.Series(portfolio_curve)
    final_val   = capital
    total_return = (final_val / initial_capital - 1) * 100
    returns      = curve.pct_change().dropna()
    sharpe       = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0
    peak         = curve.cummax()
    max_dd       = float(((curve - peak) / peak).min() * 100)

    win_trades   = [t for t in trades if t["Getiri %"] > 0]
    win_rate     = len(win_trades) / len(trades) * 100 if trades else 0

    if progress_callback:
        progress_callback(1.0, "Tamamlandı")

    return {
        "portfolio_curve":  curve,
        "trades":           trades,
        "daily_positions":  pd.DataFrame(daily_positions),
        "metrics": {
            "Toplam Getiri (%)":  round(total_return, 2),
            "Sharpe":             round(sharpe, 3),
            "Max Düşüş (%)":      round(max_dd, 2),
            "İşlem Sayısı":       len(trades),
            "Kazanma (%)":        round(win_rate, 1),
            "Final Portföy":      round(final_val, 2),
        },
    }
