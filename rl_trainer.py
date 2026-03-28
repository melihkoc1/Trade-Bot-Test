"""
RL Trainer — PPO ile BIST50 hisselerinde model eğitimi.

Çalıştırma:
    python rl_trainer.py                    # tüm BIST50, 1M timestep
    python rl_trainer.py --timesteps 500000 # hızlı test
    python rl_trainer.py --symbols AKBNK GARAN # sadece bu hisseler

Model kaydedilir: models/ppo_tradebot.zip
"""

import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import yfinance as yf

from config import BIST50_SYMBOLS
from rl_environment import BISTTradingEnv

MODEL_PATH = "models/ppo_tradebot"
VECNORM_PATH = "models/ppo_vecnorm.pkl"


def _fetch(symbol: str, period: str = "5y") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d")
        if df is None or df.empty or len(df) < 250:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df["Volume"] = df["Volume"].replace(0, 1)
        return df
    except Exception as e:
        print(f"[HATA] {symbol}: {e}")
        return None


def _fetch_bist100_returns(period: str = "3y") -> pd.Series:
    """XU100.IS endeks verisini al, 5 günlük getiri serisi döndür."""
    try:
        df = yf.Ticker("XU100.IS").history(period=period, interval="1d")
        if df is None or df.empty:
            return None
        return df["Close"].pct_change(5)
    except Exception:
        return None


def make_env(df: pd.DataFrame, bist100: pd.Series, train: bool = True):
    """Train/test split yaparak environment factory döndürür."""
    split = int(len(df) * 0.65)
    data = df.iloc[:split] if train else df.iloc[split:]
    if len(data) < 210:
        return None

    def _make():
        return BISTTradingEnv(data.copy(), bist100)

    return _make


def build_vec_env(symbols, bist100, train=True):
    """Tüm hisselerden VecEnv oluşturur."""
    factories = []
    for sym in symbols:
        df = _fetch(sym)
        if df is None:
            print(f"[SKIP] {sym} — yetersiz veri")
            continue
        factory = make_env(df, bist100, train=train)
        if factory is None:
            print(f"[SKIP] {sym} — split sonrası yetersiz veri")
            continue
        factories.append(factory)
        print(f"[OK] {sym} eklendi ({len(df)} bar)")

    if not factories:
        raise RuntimeError("Hiç geçerli hisse bulunamadı!")

    vec = DummyVecEnv(factories)
    return vec


def train(symbols=None, timesteps=1_000_000):
    os.makedirs("models", exist_ok=True)

    if symbols is None:
        symbols = BIST50_SYMBOLS

    print(f"\n{'='*50}")
    print(f"Eğitim başlıyor — {len(symbols)} hisse, {timesteps:,} timestep")
    print(f"{'='*50}\n")

    bist100 = _fetch_bist100_returns()

    # ── Train ortamı ─────────────────────────────────────────────────
    print("Veri çekiliyor (train)...")
    train_vec = build_vec_env(symbols, bist100, train=True)
    train_vec = VecNormalize(train_vec, norm_obs=False, norm_reward=True, clip_reward=1.0)

    # ── Eval ortamı (test split) ──────────────────────────────────────
    print("Veri çekiliyor (eval)...")
    eval_vec  = build_vec_env(symbols, bist100, train=False)
    eval_vec  = VecNormalize(eval_vec, norm_obs=False, norm_reward=False, training=False)

    # ── Model — mevcut varsa devam et, yoksa sıfırdan başla ──────────
    existing = MODEL_PATH + ".zip"
    if os.path.exists(existing):
        print(f"Mevcut model yükleniyor: {existing}")
        model = PPO.load(existing, env=train_vec, device="cpu")
        model.learning_rate = 5e-5  # fine-tune: yüksek lr modeli "unutturur"
    else:
        print("Yeni model olusturuluyor...")
        model = PPO(
            "MlpPolicy",
            train_vec,
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256, 128]},
            verbose=1,
        )

    callbacks = [
        EvalCallback(
            eval_vec,
            best_model_save_path="models/",
            log_path="models/logs/",
            eval_freq=max(50_000 // len(symbols), 5000),
            n_eval_episodes=len(symbols),
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // len(symbols), 10000),
            save_path="models/checkpoints/",
            name_prefix="ppo_tradebot",
        ),
    ]

    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    # ── Kaydet ───────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    train_vec.save(VECNORM_PATH)
    print(f"\n[TAMAM] Model kaydedildi: {MODEL_PATH}.zip")
    print(f"[TAMAM] VecNorm kaydedildi: {VECNORM_PATH}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Tradebot Eğitici")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Sadece belirli hisseler (ör: AKBNK GARAN)")
    args = parser.parse_args()

    syms = [f"{s}.IS" if not s.endswith(".IS") else s for s in args.symbols] \
           if args.symbols else None

    train(symbols=syms, timesteps=args.timesteps)
