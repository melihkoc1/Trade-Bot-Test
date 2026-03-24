"""
RL Trading Environment — BIST50 için OpenAI Gym uyumlu ortam.

State (18 feature, normalize edilmiş):
  Trend    : price/sma50, price/sma200, sma50/sma200, supertrend_dir, adx
  Momentum : rsi, macd_hist/atr, return_1d, return_5d, return_20d
  Volatilite: atr/price, bb_width/price
  Hacim    : volume_ratio
  Pozisyon : in_position, unrealized_pnl, days_held, drawdown_from_peak
  Piyasa   : bist100_return_5d

Action: 0=BEKLE  1=AL  2=SAT
Reward: günlük getiri − işlem maliyeti − drawdown cezası
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_atr, calculate_adx,
    calculate_supertrend,
)

TRANSACTION_COST  = 0.001   # %0.1 alış + satış komisyonu
DRAWDOWN_PENALTY  = 0.002   # portföy peak'inden %5+ düşüş cezası
POS_DD_PENALTY    = 0.004   # pozisyon içi peak'ten %8+ düşüş cezası (trailing stop öğretir)
MIN_HOLD_DAYS     = 10      # erken satış cezası eşiği (bar sayısı)
EARLY_SELL_PENALTY = 0.005  # 10 günden önce satışa ceza


def _compute_features(df: pd.DataFrame, bist100_returns: pd.Series = None) -> pd.DataFrame:
    """
    Ham OHLCV verisinden normalize feature DataFrame'i üretir.
    Her satır bir bar'ı temsil eder.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    sma50  = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    rsi    = calculate_rsi(close, 14)
    _, _, macd_hist = calculate_macd(close)
    atr    = calculate_atr(high, low, close, 14)
    adx_result = calculate_adx(high, low, close, 14)  # returns (adx, plus_di, minus_di)
    st_result  = calculate_supertrend(df, length=7, multiplier=3)

    # ── ADX sütununu al ──────────────────────────────────────────────
    if isinstance(adx_result, tuple):
        adx = adx_result[0]   # first element is ADX
    elif isinstance(adx_result, pd.DataFrame):
        adx = adx_result["ADX"] if "ADX" in adx_result.columns else adx_result.iloc[:, 0]
    else:
        adx = adx_result

    # ── SuperTrend yönü ─────────────────────────────────────────────
    # calculate_supertrend returns (values_series, direction_series) tuple
    if isinstance(st_result, tuple):
        st_dir = st_result[1].fillna(0.0)   # direction: -1 / +1
    elif isinstance(st_result, pd.DataFrame):
        if "SuperTrend_Dir" in st_result.columns:
            st_dir = st_result["SuperTrend_Dir"]
        elif "Direction" in st_result.columns:
            st_dir = st_result["Direction"]
        else:
            st_val = st_result.iloc[:, 0]
            st_dir = pd.Series(np.where(close > st_val, 1.0, -1.0), index=df.index)
    else:
        st_dir = pd.Series(np.where(close > st_result, 1.0, -1.0), index=df.index)

    # ── Bollinger band genişliği ─────────────────────────────────────
    sma20    = calculate_sma(close, 20)
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / close

    # ── Hacim oranı ──────────────────────────────────────────────────
    vol_ma20   = volume.rolling(20).mean()
    volume_ratio = volume / vol_ma20.replace(0, np.nan)

    # ── Getiriler ────────────────────────────────────────────────────
    ret1  = close.pct_change(1)
    ret5  = close.pct_change(5)
    ret20 = close.pct_change(20)

    # ── Piyasa bağlamı ───────────────────────────────────────────────
    if bist100_returns is not None:
        bist_ret5 = bist100_returns.reindex(df.index, method="ffill").fillna(0)
    else:
        bist_ret5 = pd.Series(0.0, index=df.index)

    feat = pd.DataFrame({
        # Trend (5)
        "f_price_sma50":   (close / sma50  - 1).clip(-0.5, 0.5),
        "f_price_sma200":  (close / sma200 - 1).clip(-0.5, 0.5),
        "f_sma50_sma200":  (sma50  / sma200 - 1).clip(-0.5, 0.5),
        "f_st_dir":        st_dir.astype(float),          # -1 veya +1
        "f_adx":           (adx / 100.0).clip(0, 1),

        # Momentum (5)
        "f_rsi":           (rsi / 100.0).clip(0, 1),
        "f_macd_atr":      (macd_hist / atr.replace(0, np.nan)).clip(-3, 3) / 3,
        "f_ret1":          ret1.clip(-0.1, 0.1) / 0.1,
        "f_ret5":          ret5.clip(-0.2, 0.2) / 0.2,
        "f_ret20":         ret20.clip(-0.4, 0.4) / 0.4,

        # Volatilite (2)
        "f_atr_price":     (atr / close).clip(0, 0.1) / 0.1,
        "f_bb_width":      bb_width.clip(0, 0.2) / 0.2,

        # Hacim (1)
        "f_vol_ratio":     volume_ratio.clip(0, 5) / 5,

        # Piyasa (1)
        "f_bist_ret5":     bist_ret5.clip(-0.15, 0.15) / 0.15,
    }, index=df.index)

    # Pozisyon state'leri agent tarafından eklenir (in_position, pnl, days, dd)
    return feat.fillna(0.0)


N_FEATURES = 14 + 5  # 14 piyasa feature + 5 pozisyon feature


class BISTTradingEnv(gym.Env):
    """
    Tek hisse için günlük trading environment.

    Kullanım:
        env = BISTTradingEnv(df)            # ham OHLCV DataFrame
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, bist100_returns: pd.Series = None,
                 initial_capital: float = 100_000.0):
        super().__init__()

        self.initial_capital = initial_capital
        self.features = _compute_features(df, bist100_returns)
        self.close = df["Close"].values
        self.n_steps = len(self.features)

        # Geçerli satır sayısı — ilk 200 bar warmup için boş olabilir
        self.valid_start = int(self.features.isnull().any(axis=1).sum())
        self.valid_start = max(self.valid_start, 200)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_FEATURES,), dtype=np.float32
        )
        # 0=BEKLE  1=AL  2=SAT
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    # ────────────────────────────────────────────────────────────────
    def _reset_state(self):
        self.current_step   = self.valid_start
        self.capital        = self.initial_capital
        self.shares         = 0.0
        self.buy_price      = 0.0
        self.days_held      = 0
        self.peak_value     = self.initial_capital
        self.portfolio_val  = self.initial_capital
        self.position_peak  = 0.0   # pozisyon içindeyken görülen en yüksek fiyat

    def _portfolio_value(self):
        price = self.close[self.current_step]
        return self.capital + self.shares * price

    def _obs(self):
        row = self.features.iloc[self.current_step].values.astype(np.float32)

        price = self.close[self.current_step]
        in_pos = 1.0 if self.shares > 0 else 0.0
        unreal_pnl = ((price - self.buy_price) / self.buy_price) if self.shares > 0 else 0.0
        days_norm   = min(self.days_held / 60.0, 1.0)
        pv          = self._portfolio_value()
        drawdown    = max(0.0, (self.peak_value - pv) / self.peak_value)
        # Pozisyon içi peak'ten ne kadar düştük (trailing stop sinyali)
        pos_dd = (max(0.0, (self.position_peak - price) / self.position_peak)
                  if self.shares > 0 and self.position_peak > 0 else 0.0)

        pos_features = np.array([
            in_pos,
            np.clip(unreal_pnl, -0.5, 0.5),
            days_norm,
            np.clip(drawdown, 0.0, 0.5) * 2,
            np.clip(pos_dd, 0.0, 0.3) / 0.3,   # 0-1: pozisyon peak'inden düşüş
        ], dtype=np.float32)

        return np.concatenate([row, pos_features])

    # ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        price     = self.close[self.current_step]
        prev_val  = self._portfolio_value()

        # ── Aksiyon uygula ──────────────────────────────────────────
        cost = 0.0

        if action == 1 and self.shares == 0:          # AL
            affordable = self.capital * (1 - TRANSACTION_COST)
            self.shares       = affordable / price
            self.capital      = 0.0
            self.buy_price    = price
            self.position_peak = price
            self.days_held    = 0
            cost = self.initial_capital * TRANSACTION_COST

        elif action == 2 and self.shares > 0:         # SAT
            revenue      = self.shares * price * (1 - TRANSACTION_COST)
            self.capital = revenue
            early_sell   = self.days_held < MIN_HOLD_DAYS  # erken satış flag
            self.shares  = 0.0
            self.buy_price    = 0.0
            self.position_peak = 0.0
            cost = self.initial_capital * TRANSACTION_COST
            if early_sell:
                cost += self.initial_capital * EARLY_SELL_PENALTY

        if self.shares > 0:
            self.days_held += 1
            self.position_peak = max(self.position_peak, price)

        # ── Sonraki adıma geç ───────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        # ── Ödül hesapla ─────────────────────────────────────────────
        cur_price = self.close[self.current_step]
        new_val = self._portfolio_value()
        self.peak_value = max(self.peak_value, new_val)

        pct_return = (new_val - prev_val) / prev_val

        # Benchmark (buy & hold) günlük getirisi — modelin yenmesi gereken
        bh_daily = (cur_price - price) / price
        excess_return = pct_return - bh_daily   # pozitif = piyasayı yendik

        # Portföy seviyesi drawdown cezası
        drawdown   = max(0.0, (self.peak_value - new_val) / self.peak_value)
        dd_penalty = DRAWDOWN_PENALTY * max(0.0, drawdown - 0.05)

        # Pozisyon içi trailing-stop cezası: peak'ten %8+ düşünce ağır ceza
        if self.shares > 0 and self.position_peak > 0:
            pos_dd = max(0.0, (self.position_peak - cur_price) / self.position_peak)
            pos_dd_penalty = POS_DD_PENALTY * max(0.0, pos_dd - 0.08)
        else:
            pos_dd_penalty = 0.0

        reward = 5.0 * excess_return - dd_penalty - pos_dd_penalty - (cost / self.initial_capital)
        reward = float(np.clip(reward, -1.0, 1.0))

        self.portfolio_val = new_val

        return self._obs(), reward, terminated, False, {
            "portfolio_value": new_val,
            "drawdown": drawdown,
        }

    def render(self):
        pass
