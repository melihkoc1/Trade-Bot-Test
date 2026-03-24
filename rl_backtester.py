"""
RL Backtester — Eğitilmiş PPO modelini tek hisse üzerinde test eder.

Döndürür:
    dict — getiri, trade listesi, portfolio curve, kural tabanlı ile karşılaştırma
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_environment import BISTTradingEnv

MODEL_PATH  = "models/ppo_tradebot.zip"
BEST_PATH   = "models/best_model.zip"
VECNORM_PATH = "models/ppo_vecnorm.pkl"

COMMISSION = 0.001   # %0.1


def _load_model():
    """En iyi modeli yükle (önce best_model, yoksa son model)."""
    path = BEST_PATH if os.path.exists(BEST_PATH) else MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Eğitilmiş model bulunamadı: {path}\n"
            "Önce 'python rl_trainer.py' ile modeli eğitin."
        )
    return PPO.load(path)


def run_rl_backtest(
    symbol: str,
    period: str = "2y",
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Eğitilmiş RL ajanını verilen hisse ve periyotta çalıştırır.

    Returns
    -------
    dict:
        symbol, total_return_pct, n_trades, win_rate, max_drawdown,
        sharpe, trades (list), portfolio_curve (pd.Series), status
    """
    # ── Veri çek ─────────────────────────────────────────────────────
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df is None or df.empty or len(df) < 60:
            return {"symbol": symbol, "status": "veri_yok"}
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df["Volume"] = df["Volume"].replace(0, 1)
    except Exception as e:
        return {"symbol": symbol, "status": f"hata: {e}"}

    # ── Piyasa verisi ─────────────────────────────────────────────────
    try:
        bist_df = yf.Ticker("XU100.IS").history(period=period, interval="1d")
        bist_ret5 = bist_df["Close"].pct_change(5) if not bist_df.empty else None
    except Exception:
        bist_ret5 = None

    # ── Model yükle ───────────────────────────────────────────────────
    try:
        model = _load_model()
    except FileNotFoundError as e:
        return {"symbol": symbol, "status": str(e)}

    # ── Ortam kur ─────────────────────────────────────────────────────
    env = BISTTradingEnv(df.copy(), bist100_returns=bist_ret5,
                          initial_capital=initial_capital)
    obs, _ = env.reset()

    # ── Simülasyon ────────────────────────────────────────────────────
    capital       = initial_capital
    shares        = 0.0
    buy_price     = 0.0
    trades        = []
    portfolio_curve = []
    dates         = df.index[env.valid_start:]

    step_idx  = 0
    buy_date  = None
    done      = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        price     = env.close[env.current_step]
        cur_date  = dates[step_idx] if step_idx < len(dates) else None

        # Kendi portföy takibimizi paralel yürüt (RL env kendi hesaplar)
        if action == 1 and shares == 0:           # AL
            shares    = capital * (1 - COMMISSION) / price
            buy_price = price
            buy_date  = cur_date
            capital   = 0.0
        elif action == 2 and shares > 0:          # SAT
            revenue  = shares * price * (1 - COMMISSION)
            pnl_pct  = (price - buy_price) / buy_price * 100
            trades.append({
                "type":       "SAT",
                "buy_price":  round(buy_price, 2),
                "sell_price": round(price, 2),
                "pnl_pct":    round(pnl_pct, 2),
                "buy_date":   buy_date,
                "sell_date":  cur_date,
            })
            capital   = revenue
            shares    = 0.0
            buy_price = 0.0
            buy_date  = None

        pv = capital + shares * price
        portfolio_curve.append(pv)

        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        step_idx += 1

    # Son açık pozisyonu kapat
    if shares > 0:
        price   = env.close[min(env.current_step, len(env.close) - 1)]
        revenue = shares * price * (1 - COMMISSION)
        pnl_pct = (price - buy_price) / buy_price * 100
        trades.append({
            "type":       "SAT(son)",
            "buy_price":  round(buy_price, 2),
            "sell_price": round(price, 2),
            "pnl_pct":    round(pnl_pct, 2),
            "buy_date":   buy_date,
            "sell_date":  None,
        })
        capital = revenue
        shares  = 0.0
        portfolio_curve.append(capital)   # son kapanış değerini ekle

    # ── Metrikler — portfolio_curve üzerinden hesapla ─────────────────
    curve = pd.Series(portfolio_curve)
    final_val    = float(curve.iloc[-1]) if len(curve) > 0 else initial_capital
    total_return = (final_val - initial_capital) / initial_capital * 100
    peak  = curve.cummax()
    dd    = ((curve - peak) / peak)
    max_dd = float(dd.min() * 100)

    pnl_list = [t["pnl_pct"] for t in trades]
    n_trades  = len(pnl_list)
    win_rate  = (sum(p > 0 for p in pnl_list) / n_trades * 100) if n_trades else 0

    daily_returns = curve.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)

    # ── Portfolio curve — tarihli index ──────────────────────────────
    curve_dates = dates[:len(portfolio_curve)]
    portfolio_series = pd.Series(portfolio_curve, index=curve_dates, name="RL Portfolio")

    # ── Buy & Hold karşılaştırması ───────────────────────────────────
    bh_start = df["Close"].iloc[env.valid_start]
    bh_end   = df["Close"].iloc[-1]
    bh_return = (bh_end - bh_start) / bh_start * 100

    return {
        "symbol":          symbol,
        "status":          "ok",
        "total_return_pct": round(total_return, 2),
        "buy_and_hold_pct": round(bh_return, 2),
        "n_trades":        n_trades,
        "win_rate":        round(win_rate, 1),
        "max_drawdown":    round(max_dd, 2),
        "sharpe":          round(float(sharpe), 2),
        "trades":          trades,
        "portfolio_curve": portfolio_series,
    }
