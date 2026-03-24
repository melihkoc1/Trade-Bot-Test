import numpy as np
from data_fetcher import fetch_stock_data
from indicators import add_technical_indicators, detect_divergences
from strategy import analyze_single_stock


def _calc_sharpe(equity_curve, periods_per_year=252):
    """Yillik Sharpe oranini hesapla (risksiz faiz = 0)."""
    eq = np.array(equity_curve, dtype=float)
    returns = np.diff(eq) / eq[:-1]
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _calc_max_drawdown(equity_curve):
    """Maksimum dusus (%) hesapla."""
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min() * 100)

def run_backtest(symbol, period="1y", interval="1d", initial_capital=100000,
                 stop_loss_pct=-7, take_profit_pct=10, use_strategy_sync=True,
                 use_trailing_stop=True, trailing_pct=5,
                 use_atr_trailing=True, use_volume_peak=True,
                 profile_name="Trend Avcisi",
                 exit_strategy="full"):
    """
    Backtest Motoru.
    exit_strategy:
      "full"       — Tüm pozisyon trailing stop veya SAT sinyalinde çıkar (varsayılan)
      "partial_2r" — %50 pozisyon 2R hedefinde çıkar, kalan %50 breakeven stop ile devam eder
    """
    
    df = fetch_stock_data(symbol, period=period, interval=interval)
    if df is None or df.empty:
        return None
    
    df = add_technical_indicators(df)
    df = detect_divergences(df, left_bars=5, right_bars=5)
    df = df.dropna(subset=["RSI_14", "MACD_Hist", "SMA_21"])
    
    if len(df) < 50: # Daha fazla veri lazım (SMA200 vb. için)
        return None
    
    trades = []
    capital = initial_capital
    position = None
    equity_curve = []

    max_price_since_buy = 0
    last_sell_bar = -10  # Re-entry cooldown: son satıştan 5 bar bekle
    
    for i in range(len(df)):
        price = df.iloc[i]["Close"]
        vol = df.iloc[i]["Volume"]
        current_equity = capital
        if position is not None:
             current_equity += position["shares"] * price
        equity_curve.append(current_equity)

        if i < 30: continue # Isınma periyodu

        date = df.index[i]
        df_slice = df.iloc[:i+1] # Sadece geçmişe bak
        
        # Strateji Puanını Al
        if use_strategy_sync:
            analysis = analyze_single_stock(symbol, df=df_slice, profile_name=profile_name, is_backtest=True)
            score = analysis["score"]
            signal = analysis["signal"]
        else:
            score = 55 if df.iloc[i]["RSI_14"] < 40 else 40
            signal = "AL" if score >= 50 else "BEKLE"

        # ---- ALIŞ KOŞULU ----
        if position is None:
            cooldown_ok = (i - last_sell_bar) >= 5
            if signal in ["AL", "GUCLU AL"] and score >= 55 and cooldown_ok:
                shares = int(capital * 0.95 / price)
                if shares > 0:
                    cost = shares * price
                    stop_price = price * (1 + stop_loss_pct / 100)
                    # 2R hedefi: giriş riskinin 2 katı kadar yukarı
                    initial_risk = price - stop_price
                    position = {
                        "buy_date": date,
                        "buy_price": price,
                        "shares": shares,
                        "cost": cost,
                        "stop_price": stop_price,
                        "target_2r": price + 2 * initial_risk,
                        "half_sold": False,
                    }
                    max_price_since_buy = price
                    capital -= cost
        
        # ---- SATIŞ KOŞULU (Trailing Stop Veya Sinyal) ----
        elif position is not None:
            max_price_since_buy = max(max_price_since_buy, price)

            # ---- KISMİ ÇIKIŞ: %50 pozisyon 2R hedefinde sat ----
            if exit_strategy == "partial_2r" and not position["half_sold"]:
                if price >= position["target_2r"]:
                    half = position["shares"] // 2
                    if half > 0:
                        revenue = half * price
                        partial_profit = revenue - half * position["buy_price"]
                        capital += revenue
                        position["shares"] -= half
                        position["half_sold"] = True
                        # Kalan yarı için stop'u breakeven'a taşı
                        position["stop_price"] = position["buy_price"]
                        trades.append({
                            "symbol": symbol,
                            "buy_date": position["buy_date"].strftime("%Y-%m-%d"),
                            "buy_price": round(position["buy_price"], 2),
                            "sell_date": date.strftime("%Y-%m-%d"),
                            "sell_price": round(price, 2),
                            "profit": round(partial_profit, 2),
                            "profit_pct": round((price / position["buy_price"] - 1) * 100, 2),
                            "reason": "Kısmi Çıkış — 2R Hedefi (%50)"
                        })
            
            # --- Dinamik Stop Modülleri (Elite Ultra) ---
            current_trailing_pct = trailing_pct
            
            # 1. Volume Blow-off Top (Hacimli Tepe) Filtresi
            if use_volume_peak:
                avg_vol = df["Volume"].iloc[max(0, i-20):i].mean()
                is_high_price = price >= df["Close"].iloc[max(0, i-20):i].max()
                if is_high_price and vol > (avg_vol * 2.5):
                    # Stop'u %5'e daralt (eskisi %1'di — çok agresifti, erken çıkış yapıyordu)
                    current_trailing_pct = 5.0
                    sell_reason_prefix = "Hacimli Tepe | "
                else:
                    sell_reason_prefix = ""
            else:
                sell_reason_prefix = ""

            # 2. Dynamic ATR Trailing (çarpan 2.5 → 3.5: trendin nefes almasına izin ver)
            if use_atr_trailing:
                atr = df.iloc[i]["ATR_14"]
                atr_stop_pct = (atr * 3.5 / price) * 100
                # Min %3 (gürültü eşiği), max %15 (felaket koruması)
                current_trailing_pct = max(3.0, min(15.0, atr_stop_pct))
                if sell_reason_prefix == "": sell_reason_prefix = "Dinamik ATR | "

            # Trailing Stop Uygula
            if use_trailing_stop:
                dynamic_stop = max_price_since_buy * (1 - current_trailing_pct / 100)
                is_stop_hit = price <= dynamic_stop
                sell_reason = f"{sell_reason_prefix}Trailing Stop (%{current_trailing_pct:.1f})"
            else:
                change_pct = (price - position["buy_price"]) / position["buy_price"] * 100
                is_stop_hit = change_pct <= stop_loss_pct or change_pct >= take_profit_pct
                sell_reason = "Sabit Stop/Kar Al"
            
            # Ya da strateji GÜÇLÜ SAT verirse çık
            is_strategy_sell = signal == "SAT" or score <= 25
            
            if is_stop_hit or is_strategy_sell:
                revenue = position["shares"] * price
                profit = revenue - position["cost"]
                capital += revenue
                
                trades.append({
                    "symbol": symbol,
                    "buy_date": position["buy_date"].strftime("%Y-%m-%d"),
                    "buy_price": round(position["buy_price"], 2),
                    "sell_date": date.strftime("%Y-%m-%d"),
                    "sell_price": round(price, 2),
                    "profit": round(profit, 2),
                    "profit_pct": round(((price - position["buy_price"]) / position["buy_price"] * 100), 2),
                    "reason": sell_reason if is_stop_hit else "Strateji SAT"
                })
                position = None
                last_sell_bar = i
    
    # Rapor Taslağı
    df["Equity"] = equity_curve
    current_total = capital
    if position is not None:
        current_total = capital + position["shares"] * df.iloc[-1]["Close"]
    
    total_profit = current_total - initial_capital

    sharpe = _calc_sharpe(equity_curve)
    max_dd = _calc_max_drawdown(equity_curve)

    report = {
        "symbol": symbol,
        "profile": profile_name,
        "final_capital": round(current_total, 2),
        "total_profit": round(total_profit, 2),
        "total_return_pct": round((total_profit / initial_capital) * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(trades),
        "winning_trades": len([t for t in trades if t["profit"] > 0]),
        "losing_trades": len([t for t in trades if t["profit"] <= 0]),
        "win_rate": round(len([t for t in trades if t["profit"] > 0]) / len(trades) * 100, 1) if trades else 0,
        "trades": trades,
        "df": df
    }
    return report

def print_report(report):
    """Backtest sonuç raporunu terminale yazdırır."""
    if report is None:
        return
    
    print(f"\n{'='*55}")
    print(f"  BACKTEST RAPORU: {report['symbol']}")
    print(f"{'='*55}")
    print(f"  Periyot:          {report['period']} ({report['data_points']} mum)")
    print(f"  Başlangıç:        {report['initial_capital']:,.0f} ₺")
    print(f"  Son Durum:        {report['final_capital']:,.0f} ₺")
    
    color = "+" if report['total_profit'] >= 0 else ""
    print(f"  Kâr/Zarar:        {color}{report['total_profit']:,.0f} ₺ ({color}{report['total_return_pct']:.1f}%)")
    print(f"  Toplam İşlem:     {report['total_trades']}")
    print(f"  Kazanan:          {report['winning_trades']}")
    print(f"  Kaybeden:         {report['losing_trades']}")
    print(f"  Başarı Oranı:     %{report['win_rate']}")
    
    if report["trades"]:
        print(f"\n  {'Tarih':<24} {'Alış':>8} {'Satış':>8} {'K/Z':>10} {'Sebep'}")
        print(f"  {'-'*70}")
        for t in report["trades"]:
            pnl = f"{t['profit_pct']:+.1f}%"
            indicator = "✅" if t["profit"] > 0 else "❌"
            print(f"  {indicator} {t['buy_date']} → {t['sell_date']:<12} {t['buy_price']:>8.2f} {t['sell_price']:>8.2f} {pnl:>8} {t['reason']}")
    
    print(f"{'='*55}")


if __name__ == "__main__":
    print("=" * 55)
    print("  BACKTESTING MOTORU TESTİ")
    print("=" * 55)
    
    test_symbol = "THYAO.IS"
    print(f"\n{test_symbol} üzerinde 1 yıllık backtest yapılıyor...")
    
    report = run_backtest(test_symbol, period="1y", interval="1d", initial_capital=100000)
    
    if report:
        print_report(report)
    else:
        print("Backtest başarısız.")
