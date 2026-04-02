# Tradebot v1 - Phase 29 Fixed
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from config import BIST50_SYMBOLS, SECTOR_INDEX_MAP, SECTOR_MAP
from data_fetcher import fetch_stock_data, fetch_macro_data, fetch_all_bist30
from indicators import add_technical_indicators
import strategy
import backtester 
import importlib
importlib.reload(strategy)
importlib.reload(backtester)
from backtester import run_backtest
from strategy import analyze_single_stock, scan_all_bist50, PROFILES
import portfolio_manager
import plotly.express as px
import cluster_manager
import sim_manager
import news_scraper

load_dotenv()

# ============================================================
# TEMEL ANALİZ EKRANI (Strateji dışı — sadece UI etiketi)
# ============================================================
@st.cache_data(ttl=3600)
def get_fundamental_label(symbol):
    """
    Hisse için basit temel analiz etiketi döndürür: AL / TUT / SAT
    Strateji skoruna dahil değil — sadece bilgilendirme amaçlı.
    Veri yfinance info'dan gelir (güncel, point-in-time değil).
    """
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
        pe  = info.get("trailingPE")  or info.get("forwardPE")
        pb  = info.get("priceToBook")
        de  = info.get("debtToEquity")  # Borç/Özkaynak (yüzde olarak gelir)

        flags, score = [], 0

        # Zarar eden hisse (en kritik)
        if eps is not None:
            if eps < 0:
                flags.append(f"Zarar eden (EPS {eps:.2f})")
                score -= 2

        # F/K kontrolü
        if pe is not None:
            if pe < 0:
                flags.append(f"F/K negatif")
                score -= 1
            elif pe > 40:
                flags.append(f"F/K şişik ({pe:.0f}x)")
                score -= 1

        # PD/DD kontrolü
        if pb is not None:
            if pb > 8:
                flags.append(f"PD/DD yüksek ({pb:.1f}x)")
                score -= 1

        # Borç/Özkaynak (yfinance yüzde olarak verir, 300 = 3x)
        if de is not None:
            if de > 300:
                flags.append(f"Yüksek borç ({de/100:.1f}x)")
                score -= 1

        if score <= -2:
            label = "SAT"
        elif score == -1:
            label = "TUT"
        else:
            label = "AL"

        return {
            "label": label,
            "flags": flags,
            "eps": round(eps, 2) if eps else None,
            "pe":  round(pe, 1)  if pe  else None,
            "pb":  round(pb, 1)  if pb  else None,
        }
    except Exception:
        return {"label": "?", "flags": [], "eps": None, "pe": None, "pb": None}


def _fundamental_badge(label):
    """Temel analiz etiketi için renkli HTML badge."""
    colors = {"AL": "#00c853", "TUT": "#ffa000", "SAT": "#d32f2f", "?": "#555"}
    c = colors.get(label, "#555")
    return f'<span style="background:{c};color:#fff;border-radius:4px;padding:1px 7px;font-size:0.75em;font-weight:bold;margin-left:6px;">Temel: {label}</span>'


# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="BIST50 Tradebot v1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    .signal-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #444; backdrop-filter: blur(10px);
    }
    .signal-al { border-left-color: #00e676 !important; }
    .signal-guclu-al { border-left-color: #00ff88 !important; background: rgba(0,230,118,0.08) !important; }
    .signal-incele { border-left-color: #ffc107 !important; }
    .signal-sat { border-left-color: #ff1744 !important; }
    .signal-dikkat { border-left-color: #ff9100 !important; }
    .signal-bekle { border-left-color: #546e7a !important; }
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #00e676, #00b0ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 32px; font-weight: 800; margin-bottom: 5px;
    }
    .sub-title { text-align: center; color: #666; font-size: 14px; margin-bottom: 20px; }
    .profile-box {
        background: rgba(0,176,255,0.08); border: 1px solid rgba(0,176,255,0.2);
        border-radius: 8px; padding: 10px 14px; margin: 8px 0; font-size: 12px; color: #aaa;
    }
    .news-card {
        background: rgba(255,255,255,0.03); border-radius: 8px;
        padding: 12px 16px; margin: 6px 0; border-left: 3px solid #00b0ff;
    }
    .news-title { color: #ccc; font-size: 14px; font-weight: 500; }
    .news-meta { color: #666; font-size: 11px; margin-top: 4px; }
    .ai-card {
        background: linear-gradient(45deg, rgba(0,230,118,0.1), rgba(0,176,255,0.1));
        border: 1px solid rgba(0,176,255,0.3);
        border-radius: 12px; padding: 15px; margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BASLIK
# ============================================================
st.markdown('<p class="main-title">📈 BIST30 Tradebot v1</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Teknik Analiz & Hacim & Divergence & Haber</p>', unsafe_allow_html=True)

# ============================================================
# SOL PANEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    scan_mode_sidebar = st.radio(
        "Ek Modlar",
        ["Normal Analiz", "Backtesting"],
        index=0
    )
    
    st.divider()
    
    if scan_mode_sidebar == "Backtesting":
        bt_profile = st.selectbox("Yatirim Profili", ["Trend Avcisi", "Deger Yatirimcisi"], index=0)
        bt_p = PROFILES[bt_profile]
        bt_symbol = st.selectbox("Backtest Sembol", BIST50_SYMBOLS, index=0)
        bt_period = st.selectbox("Backtest Suresi", ["3mo", "6mo", "1y", "2y"], index=2)
        bt_capital = st.number_input("Baslangic Sermayesi", value=100000, step=10000)
        bt_stop_loss = st.slider(
            "Stop-Loss (%)", min_value=-30, max_value=-3, 
            value=bt_p["stop_loss"], step=1,
            help="Zarar bu yuzdeye ulasinca pozisyon kapatilir"
        )
        bt_take_profit = st.slider(
            "Kar Hedefi (%)", min_value=3, max_value=50, 
            value=bt_p["take_profit"], step=1,
            help="Kar bu yuzdeye ulasinca pozisyon kapatilir"
        )
        
        # Faz 30: Elite Quant Ayarları
        st.divider()
        st.caption("🛡️ Elite Quant Gelişmiş Ayarlar")
        bt_strategy_sync = st.toggle("Strateji Senkronizasyonu (38 Kural)", value=True)
        bt_use_trailing = st.toggle("İz Süren Stop (Trailing Stop)", value=True)
        bt_trailing_pct = st.slider("Trailing Tabanı (%)", 2, 15, 5, help="Zirveden ne kadar dusunce satilsin?")

        # Çıkış Stratejisi
        bt_exit_strategy = st.selectbox(
            "Çıkış Stratejisi",
            options=["full", "partial_2r"],
            format_func=lambda x: {
                "full": "Tam Çıkış — Tüm pozisyon trailing stop ile çıkar",
                "partial_2r": "Kısmi Çıkış — %50 → 2R'de sat, %50 → trailing devam"
            }[x],
            help="partial_2r: İlk yarı 2× risk mesafesinde satılır, stop breakeven'a taşınır."
        )

        # Elite Ultra (Faz 31)
        c1, c2 = st.columns(2)
        bt_use_atr = c1.toggle("Dinamik ATR Takibi", value=True, help="Stop mesafesini oynakliga (ATR) gore ayarlar")
        bt_use_vol_peak = c2.toggle("Hacimli Zirve (Blow-off) Koruma", value=True, help="Zirvede asiri hacim gorulurse kapiyi daraltir")
        st.markdown(f"""
        <div class="profile-box">
            📋 Stop-Loss: {bt_stop_loss}%<br>
            🎯 Kar Hedefi: {bt_take_profit}%
        </div>
        """, unsafe_allow_html=True)
    else:
        profile_names = list(PROFILES.keys())
        profile_name = st.selectbox(
            "Yatirim Profili",
            profile_names,
            index=0,
            help="Yatirim tarzina gore ayarlar otomatik belirlenir"
        )
        
        selected_profile = PROFILES[profile_name]
        
        if profile_name == "Manuel":
            period = st.selectbox("Veri Suresi", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"], index=4)
            interval = st.selectbox("Mum Periyodu", ["5m", "15m", "30m", "1h", "1d", "1wk"], index=4)
        else:
            period = selected_profile["period"]
            interval = selected_profile["interval"]
        
        st.markdown(f"""
        <div class="profile-box">
            {selected_profile['description']}<br>
            📊 {period} veri, {interval} mum<br>
            🎯 Kar: %{selected_profile['take_profit']} | Stop: %{selected_profile['stop_loss']}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.caption("🐢 Turtle Risk Yonetimi (Lot Hesabi)")
        invest_capital = st.number_input("Sermaye (TL)", value=100000, step=10000)
        risk_pct = st.slider("Maks. Risk (%)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        turtle_active = st.checkbox("Kaplumbaga (Donchian) Stratejisini Etkinlestir", value=False)
        
        st.divider()
    st.divider()
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_ready = bool(telegram_token and telegram_chat_id)
    send_notification = False
    if scan_mode_sidebar != "Backtesting" and telegram_ready:
        send_notification = st.checkbox("Telegram'a gonder", value=False)
    
    start_scan = st.button("🚀 Baslat", use_container_width=True, type="primary")

# ============================================================
# YARDIMCI GORSELLESTIRME
# ============================================================
def get_signal_style(signal):
    styles = {
        "GUCLU AL": ("signal-guclu-al", "🟢🟢"),
        "AL": ("signal-al", "🟢"),
        "INCELE": ("signal-incele", "🟡"),
        "SAT": ("signal-sat", "🔴"),
        "DIKKAT": ("signal-dikkat", "🟠"),
        "BEKLE": ("signal-bekle", "⚪"),
    }
    return styles.get(signal, ("signal-bekle", "⚪"))

def render_news_panel(symbol):
    st.subheader("📰 Akıllı Haber & KAP Merkezi")
    score, news = news_scraper.get_sentiment_score(symbol)
    if not news:
        st.info("Güncel haber veya KAP bildirimi bulunamadı.")
        return
    
    # Haber tipi dağılımı
    has_kap = any(n.get("type") == "KAP" for n in news)
    has_sector = any(n.get("type") == "Sektör" for n in news)
    
    if has_kap:
        st.success("🔔 Önemli: Bu hisse için yeni **KAP Bildirimleri** mevcut!")
    elif has_sector:
        st.info(f"💡 Not: Hisseye özel yeni haber yok, **Sektörel Gelişmeler** listeleniyor.")
        
    for n in news:
        sentiment_color = "#2ecc71" if n['sentiment'] == "Positive" else "#e74c3c" if n['sentiment'] == "Negative" else "#95a5a6"
        
        # Etiket Ayarları
        type_label = n.get("type", "Haber")
        label_bg = "#c0392b" if type_label == "KAP" else "#2980b9" if type_label == "Sektör" else "#2c3e50"
        
        st.markdown(f"""
        <div style="background-color: #1e272e; padding: 12px; border-radius: 8px; border-left: 5px solid {sentiment_color}; margin-bottom: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
                <span style="background-color: {label_bg}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; text-transform: uppercase;">{type_label}</span>
                <span style="font-size: 11px; color: #7f8c8d;">{n['source']}</span>
            </div>
            <div style="font-size: 14px; font-weight: bold; color: #ecf0f1; margin: 5px 0; line-height: 1.4;">{n['title']}</div>
            <div style="font-size: 12px; color: #bdc3c7; margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
                <span>🔗 <a href="{n['link']}" target="_blank" style="color: #3498db; text-decoration: none;">Detaylı Oku</a></span>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="color: {sentiment_color}; font-weight: bold;">{n['sentiment']}</span>
                    <span style="background-color: rgba(255,255,255,0.1); padding: 1px 6px; border-radius: 10px; font-size: 11px; color: #ecf0f1;">{n['score']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def _ind_badge(signal):
    """AL / SAT / NÖTR için renkli HTML badge döndürür."""
    if signal == "AL":
        return "🟢 AL"
    elif signal == "SAT":
        return "🔴 SAT"
    elif signal == "GÜÇLÜ AL":
        return "🟢🟢 GÜÇLÜ AL"
    elif signal == "GÜÇLÜ SAT":
        return "🔴🔴 GÜÇLÜ SAT"
    return "⚪ NÖTR"

def render_indicator_panel(result):
    """Tüm teknik indikatörleri AL/SAT/NÖTR sinyal tablosu olarak gösterir."""
    with st.expander("📈 Teknik İndikatör Sinyalleri", expanded=True):
        indicators = []

        # RSI
        rsi = result.get("rsi")
        if rsi is not None:
            if rsi < 30:   sig = "GÜÇLÜ AL"
            elif rsi < 40: sig = "AL"
            elif rsi > 70: sig = "GÜÇLÜ SAT"
            elif rsi > 60: sig = "SAT"
            else:          sig = "NÖTR"
            indicators.append(("RSI (14)", f"{rsi:.1f}", sig))

        # MACD
        macd = result.get("macd"); macd_sig = result.get("macd_signal")
        if macd is not None and macd_sig is not None:
            if macd > macd_sig and macd < 0:   sig = "GÜÇLÜ AL"
            elif macd > macd_sig:              sig = "AL"
            elif macd < macd_sig and macd > 0: sig = "GÜÇLÜ SAT"
            elif macd < macd_sig:              sig = "SAT"
            else:                              sig = "NÖTR"
            indicators.append(("MACD", f"{macd:.4f}", sig))

        # EMA Crossover
        ema_cross = result.get("ema_cross")
        if ema_cross:
            sig_map = {"Golden Cross": "GÜÇLÜ AL", "Yukari": "AL", "Death Cross": "GÜÇLÜ SAT", "Asagi": "SAT"}
            sig = sig_map.get(ema_cross, "NÖTR")
            indicators.append(("EMA 9/21", ema_cross, sig))

        # SMA
        price = result.get("price"); sma9 = result.get("sma_9"); sma21 = result.get("sma_21")
        if price and sma9 and sma21:
            if price > sma9 > sma21:  sig = "AL"
            elif price < sma9 < sma21: sig = "SAT"
            else:                      sig = "NÖTR"
            indicators.append(("SMA 9/21", f"{sma9:.2f} / {sma21:.2f}", sig))

        # Bollinger
        bb_lower = result.get("bb_lower"); bb_upper = result.get("bb_upper")
        if price and bb_lower and bb_upper:
            if price <= bb_lower:              sig = "GÜÇLÜ AL"
            elif price <= bb_lower * 1.02:     sig = "AL"
            elif price >= bb_upper:            sig = "GÜÇLÜ SAT"
            elif price >= bb_upper * 0.98:     sig = "SAT"
            else:                              sig = "NÖTR"
            indicators.append(("Bollinger Bantları", f"Alt:{bb_lower:.2f} Üst:{bb_upper:.2f}", sig))

        # ADX
        adx = result.get("adx"); trend_dir = result.get("trend_direction", "")
        if adx is not None:
            if adx >= 25 and trend_dir == "Yukari":   sig = "AL" if adx < 30 else "GÜÇLÜ AL"
            elif adx >= 25 and trend_dir == "Asagi":  sig = "SAT" if adx < 30 else "GÜÇLÜ SAT"
            else:                                     sig = "NÖTR"
            indicators.append(("ADX (14)", f"{adx:.1f} {trend_dir}", sig))

        # SuperTrend
        st_dir = result.get("supertrend_dir", 0)
        if st_dir != 0:
            sig = "AL" if st_dir == 1 else "SAT"
            indicators.append(("SuperTrend", "Yukari" if st_dir == 1 else "Asagi", sig))

        # Hacim
        vol_ratio = result.get("volume_ratio")
        if vol_ratio is not None:
            if vol_ratio >= 2.0:   sig = "GÜÇLÜ AL"
            elif vol_ratio >= 1.5: sig = "AL"
            elif vol_ratio < 0.5:  sig = "SAT"
            else:                  sig = "NÖTR"
            indicators.append(("Hacim Oranı", f"{vol_ratio:.2f}x", sig))

        # Divergence
        bull_div = result.get("bull_div", 0); bear_div = result.get("bear_div", 0)
        if bull_div > 0 or bear_div > 0:
            if bull_div >= 3:   sig = "GÜÇLÜ AL"
            elif bull_div >= 1: sig = "AL"
            elif bear_div >= 3: sig = "GÜÇLÜ SAT"
            elif bear_div >= 1: sig = "SAT"
            else:               sig = "NÖTR"
            indicators.append(("Divergence", f"↑{bull_div} ↓{bear_div}", sig))

        # Multi-TF
        weekly = result.get("weekly_trend")
        if weekly:
            sig = "AL" if weekly == "Yukari" else ("SAT" if weekly == "Asagi" else "NÖTR")
            indicators.append(("Haftalık Trend", weekly, sig))

        # Destek/Direnç
        sup_pct = result.get("support_distance_pct"); res_pct = result.get("resistance_distance_pct")
        if sup_pct is not None:
            if sup_pct <= 1.5:   sig = "AL"
            elif res_pct and res_pct <= 1.5: sig = "SAT"
            else:                sig = "NÖTR"
            sup_val = result.get("nearest_support", "-"); res_val = result.get("nearest_resistance", "-")
            indicators.append(("Destek/Direnç", f"D:{sup_val} R:{res_val}", sig))

        # Fibonacci
        fib_zone = result.get("fib_zone")
        if fib_zone:
            fib_bullish = any(z in str(fib_zone) for z in ["0.382", "0.5", "0.618", "Destek"])
            sig = "AL" if fib_bullish else "NÖTR"
            indicators.append(("Fibonacci", fib_zone, sig))

        # Minervini
        m_score = result.get("minervini_score", 0)
        if m_score > 0:
            sig = "GÜÇLÜ AL" if m_score >= 7 else ("AL" if m_score >= 5 else "NÖTR")
            indicators.append(("Minervini Trend", f"{int(m_score)}/8", sig))

        # RS Rating vs BIST100
        rs = result.get("rs_rating")
        if rs is not None:
            if rs >= 10:    sig = "GÜÇLÜ AL"
            elif rs >= 3:   sig = "AL"
            elif rs <= -10: sig = "GÜÇLÜ SAT"
            elif rs <= -3:  sig = "SAT"
            else:           sig = "NÖTR"
            indicators.append(("RS vs BIST100", f"{rs:+.1f}%", sig))

        # Sektör
        sector_score = result.get("sector_score", 0)
        if sector_score != 0:
            sig = "AL" if sector_score > 3 else ("SAT" if sector_score < -3 else "NÖTR")
            indicators.append(("Sektör Skoru", f"{sector_score:+.1f}", sig))

        if not indicators:
            st.info("İndikatör verisi yok.")
            return

        # Grid gösterim — 3 sütun
        cols = st.columns(3)
        for i, (name, value, sig) in enumerate(indicators):
            with cols[i % 3]:
                badge = _ind_badge(sig)
                st.markdown(f"**{name}**")
                st.markdown(f"{badge} &nbsp; `{value}`", unsafe_allow_html=False)
                st.markdown("---")

def render_single_result(result, df, capital=100000, risk_pct=2.0):
    css_class, emoji = get_signal_style(result["signal"])

    # Temel Analiz Etiketi
    fund = get_fundamental_label(result["symbol"])
    badge = _fundamental_badge(fund["label"])
    pe_str  = f"F/K: **{fund['pe']}x**"  if fund["pe"]  else "F/K: -"
    pb_str  = f"PD/DD: **{fund['pb']}x**" if fund["pb"]  else "PD/DD: -"
    eps_str = f"EPS: **{fund['eps']}**"   if fund["eps"] is not None else "EPS: -"
    flags_str = "  ·  ".join(fund["flags"]) if fund["flags"] else "Temel görünüm temiz"
    st.markdown(
        f"**Temel Analiz** {badge} &nbsp;&nbsp; {pe_str} &nbsp;|&nbsp; {pb_str} &nbsp;|&nbsp; {eps_str}"
        + (f"\n\n> {flags_str}" if fund["flags"] else ""),
        unsafe_allow_html=True
    )
    st.markdown("---")

    # İndikatör Paneli
    render_indicator_panel(result)

    # Metrikler (Üst Satır)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Fiyat", f"{result['price']} ₺")
    c2.metric("📊 RSI", f"{result['rsi']}")
    c3.metric("🎯 Skor", f"{result['score']}/100")
    c4.metric("📊 Hacim", f"{result.get('volume_ratio')}x")
    c5.metric("🔀 Div", f"↑{result.get('bull_div', 0)} ↓{result.get('bear_div', 0)}")

    # Sinyal Kartı
    st.markdown("---")
    res_col, news_col = st.columns([1, 1])
    with res_col:
        reasons_html = "".join([f"<li>{r}</li>" for r in result["reasons"]])
        st.markdown(f"""
        <div class="signal-card {css_class}">
            <h2>{emoji} {result['signal']}</h2>
            <ul>{reasons_html}</ul>
        </div>
        """, unsafe_allow_html=True)
    with news_col:
        render_news_panel(result["symbol"])

def render_backtest_results(report):
    if not report:
        st.error("Backtest sonuçları alınamadı.")
        return

    st.markdown(f"### 🧪 Backtest Sonuçları: {report['symbol']}")
    
    # Hata Kontrolü (df yoksa)
    if "df" not in report:
        st.error(f"⚠️ Kritik Hata: Backtest veri seti (df) rapor içinde bulunamadı! Mevcut veriler: {list(report.keys())}")
        st.info("İpucu: Docker kullanıyorsanız konteyneri yeniden başlatmayı veya 'backtester.py' dosyasının güncel olduğundan emin olmayı deneyin.")
        return

    # Metrikler
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Son Sermaye", f"{report['final_capital']:,} ₺")
    m2.metric("Toplam Kar", f"{report['total_profit']:,} ₺", delta=f"%{report['total_return_pct']}")
    m3.metric("İşlem Sayısı", report['total_trades'])
    m4.metric("Başarı Oranı", f"%{report['win_rate']}")

    df = report["df"]
    
    # 1. Grafik: Fiyat ve İşlemler
    st.subheader("📈 Fiyat ve Al/Sat Sinyalleri")
    fig_price = px.line(df, y="Close", title=f"{report['symbol']} Fiyat Grafiği", template="plotly_dark")
    
    # Alış ve Satış noktalarını işaretle
    buys = [t for t in report["trades"]]
    buy_dates = [t["buy_date"] for t in buys]
    buy_prices = [t["buy_price"] for t in buys]
    sell_dates = [t["sell_date"] for t in buys if "(Açık)" not in t["sell_date"]]
    sell_prices = [t["sell_price"] for t in buys if "(Açık)" not in t["sell_date"]]

    fig_price.add_scatter(x=buy_dates, y=buy_prices, mode='markers', name='ALIŞ', 
                         marker=dict(color='lime', size=12, symbol='triangle-up'))
    fig_price.add_scatter(x=sell_dates, y=sell_prices, mode='markers', name='SATIŞ', 
                         marker=dict(color='red', size=12, symbol='triangle-down'))
    
    st.plotly_chart(fig_price, use_container_width=True)

    # 2. Grafik: Sermaye Eğrisi (Equity Curve)
    st.subheader("💰 Sermaye Gelişimi (Equity Curve)")
    fig_equity = px.area(df, y="Equity", title="Toplam Portföy Değeri", template="plotly_dark", color_discrete_sequence=["#2ecc71"])
    st.plotly_chart(fig_equity, use_container_width=True)

    # 3. İşlem Listesi
    st.subheader("📝 İşlem Geçmişi")
    if report["trades"]:
        trades_df = pd.DataFrame(report["trades"])
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("Hiç işlem gerçekleşmedi.")

def render_scan_results(results):
    for r in results[:15]:
        css_class, emoji = get_signal_style(r["signal"])
        fund = get_fundamental_label(r["symbol"])
        badge = _fundamental_badge(fund["label"])
        tip = " | ".join(fund["flags"]) if fund["flags"] else "Temel görünüm temiz"
        pe_str  = f"F/K {fund['pe']}x"  if fund["pe"]  else ""
        pb_str  = f"PD/DD {fund['pb']}x" if fund["pb"]  else ""
        eps_str = f"EPS {fund['eps']}"   if fund["eps"] is not None else ""
        metrics = " · ".join(x for x in [pe_str, pb_str, eps_str] if x)
        st.markdown(f"""
        <div class="signal-card {css_class}" style="padding:10px 15px;">
            <b>{r['symbol'].replace('.IS','')}</b>: {emoji} {r['signal']} | Skor: {r['score']} | Fiyat: {r['price']} ₺
            {badge}
            <span style="color:#aaa;font-size:0.8em;margin-left:8px;">{metrics}</span>
            <span style="color:#888;font-size:0.75em;display:block;margin-top:2px;">{tip}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ANA TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📊 Tekli Hisse Analizi", "🚀 BIST50 Tam Tarama", "🗺️ Piyasa Haritası", "🎲 Risk & Simülasyon Merkezi", "🤖 RL Ajanı", "🔄 Rotasyon Stratejisi", "📋 Temel Analiz", "🧬 Kombine Strateji"])

with tab1:
    display_names = [s.replace(".IS", "") for s in BIST50_SYMBOLS]
    selected_display = st.selectbox("Hisse Secin", display_names, index=display_names.index("THYAO"))
    selected_symbol = selected_display + ".IS"
    
    if start_scan and scan_mode_sidebar == "Normal Analiz":
        with st.spinner(f"{selected_symbol} analiz ediliyor..."):
            result = analyze_single_stock(selected_symbol, period=period, interval=interval, profile_name=profile_name, turtle_active=turtle_active)
            render_single_result(result, result["df"], capital=invest_capital, risk_pct=risk_pct)

with tab2:
    if start_scan and scan_mode_sidebar == "Normal Analiz":
        with st.spinner("BIST50 taraniyor..."):
            results = scan_all_bist50(period=period, interval=interval, profile_name=profile_name)
            render_scan_results(results)
            
            # Portföy Dağılımı
            buy_signals = [r for r in results if r["signal"] in ["AL", "GUCLU AL"]]
            if buy_signals:
                st.divider()
                st.subheader("⚖️ Optimal Portföy Dağılımı")
                symbols_to_opt = [r["symbol"] for r in buy_signals[:10]]
                historical_dfs = {r["symbol"]: r["df"] for r in buy_signals if r["df"] is not None}
                expected_rets = {r["symbol"]: (r["score"]/100) for r in buy_signals}
                weights = portfolio_manager.optimize_portfolio(symbols_to_opt, expected_rets, historical_dfs)
                
                df_weights = pd.DataFrame([{"Hisse": s.replace(".IS",""), "Ağırlık (%)": w*100} for s, w in weights.items() if w > 0])
                st.plotly_chart(px.pie(df_weights, values='Ağırlık (%)', names='Hisse', title="Sermaye Dağılımı", hole=0.4), use_container_width=True)

with tab3:
    st.header("🗺️ BIST50 Piyasa Haritası (K-Means)")
    if st.button("Piyasa Haritasını Güncelle"):
        with st.spinner("Piyasa verileri işleniyor..."):
            market_data = fetch_all_bist30(period="1y", interval="1d")
            processed_data = {}
            for s, d in market_data.items():
                processed_data[s] = add_technical_indicators(d)
            
            clusters = cluster_manager.get_market_clusters(processed_data)
            if clusters is not None:
                fig = px.scatter(clusters, x="Symbol", y="Cluster_Name", color="Cluster_Name", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(clusters, use_container_width=True)

with tab4:
    st.header("🎲 Risk & Simülasyon Merkezi (Monte Carlo)")
    st.markdown("""
    Bu bölümde hisseler için **10.000+ senaryo** üzerinden gelecek 21 günlük (1 ay) fiyat projeksiyonları yapılır. 
    **Geometrik Brownian Hareketi** modeli kullanılarak piyasa volatilitesi simüle edilir.
    """)
    
    sim_display_names = [s.replace(".IS", "") for s in BIST50_SYMBOLS]
    sim_selected = st.selectbox("Simülasyon İçin Hisse Seçin", sim_display_names, index=sim_display_names.index("THYAO"))
    sim_symbol = sim_selected + ".IS"
    
    if st.button("Simülasyonu Çalıştır"):
        with st.spinner(f"{sim_symbol} için 1000 senaryo simüle ediliyor..."):
            # Veri çek
            df_sim = fetch_stock_data(sim_symbol, period="1y", interval="1d")
            if df_sim is not None:
                sim_res = sim_manager.get_monte_carlo_results(sim_symbol, df_sim, days=21)
                
                if sim_res:
                    # Metrikler
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Beklenen Fiyat (Median)", f"{sim_res['expected_price']} ₺")
                    m2.metric("Başarı Olasılığı", f"%{sim_res['success_probability']}")
                    m3.metric("%5 En Kötü Senaryo", f"{sim_res['percentile_5']} ₺")
                    m4.metric("%95 En İyi Senaryo", f"{sim_res['percentile_95']} ₺")
                    
                    # Grafik (Fan Chart)
                    st.subheader("📊 1000 Senaryo Fan Grafiği")
                    scenarios = sim_res["scenarios"]
                    # Sadece 100 senaryoyu çizelim karışıklık olmasın
                    plot_df = pd.DataFrame(scenarios[:100].T)
                    fig_sim = px.line(plot_df, template="plotly_dark", labels={"value": "Fiyat", "index": "Gün"})
                    fig_sim.update_layout(showlegend=False)
                    # Başlangıç fiyatına kırmızı çizgi
                    fig_sim.add_hline(y=sim_res["start_price"], line_dash="dash", line_color="red", annotation_text="Başlangıç")
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    # Dağılım Grafiği
                    st.subheader("🎯 Final Fiyat Dağılımı")
                    fig_dist = px.histogram(sim_res["scenarios"][:, -1], nbins=50, template="plotly_dark", title="21 Gün Sonraki Fiyat Dağılımı")
                    fig_dist.add_vline(x=sim_res["start_price"], line_color="red")
                    st.plotly_chart(fig_dist, use_container_width=True)

# Backtest sonucu — Tekli Hisse
if start_scan and scan_mode_sidebar == "Backtesting":
    from backtester import run_backtest
    with st.spinner(f"{bt_symbol} backtest yapılıyor..."):
        report = run_backtest(
            bt_symbol,
            period=bt_period,
            interval="1d",
            initial_capital=bt_capital,
            stop_loss_pct=bt_stop_loss,
            take_profit_pct=bt_take_profit,
            use_strategy_sync=bt_strategy_sync,
            use_trailing_stop=bt_use_trailing,
            trailing_pct=bt_trailing_pct,
            use_atr_trailing=bt_use_atr,
            use_volume_peak=bt_use_vol_peak,
            exit_strategy=bt_exit_strategy
        )
        render_backtest_results(report)

# ── BIST50 Multi-Hisse Backtest ──────────────────────────────
if scan_mode_sidebar == "Backtesting":
    st.divider()
    st.subheader("📊 BIST50 Toplu Backtest")
    st.caption("Tüm BIST50 hisselerine aynı strateji parametreleriyle backtest uygular. Sonuçlar stratejinin genel performansını gösterir.")

    if st.button("Tüm BIST50'yi Tara", type="primary", key="multi_bt"):
        from backtester import run_backtest
        from config import BIST50_SYMBOLS
        rows = []
        progress = st.progress(0, text="Taranıyor...")
        for idx, sym in enumerate(BIST50_SYMBOLS):
            progress.progress((idx + 1) / len(BIST50_SYMBOLS), text=f"{sym} ({idx+1}/{len(BIST50_SYMBOLS)})")
            try:
                r = run_backtest(
                    sym,
                    period=bt_period,
                    interval="1d",
                    initial_capital=bt_capital,
                    stop_loss_pct=bt_stop_loss,
                    take_profit_pct=bt_take_profit,
                    use_strategy_sync=bt_strategy_sync,
                    use_trailing_stop=bt_use_trailing,
                    trailing_pct=bt_trailing_pct,
                    use_atr_trailing=bt_use_atr,
                    use_volume_peak=bt_use_vol_peak,
                    exit_strategy=bt_exit_strategy
                )
                if r and r["total_trades"] > 0:
                    rows.append({
                        "Hisse":         sym.replace(".IS", ""),
                        "Getiri %":      r["total_return_pct"],
                        "İşlem":         r["total_trades"],
                        "Kazanan":       r["winning_trades"],
                        "Win Rate %":    r["win_rate"],
                        "Sharpe":        r["sharpe"],
                        "Max DD %":      r["max_drawdown_pct"],
                        "Son Sermaye":   r["final_capital"],
                    })
            except Exception as e:
                pass
        progress.empty()

        if rows:
            import pandas as pd
            scan_df = pd.DataFrame(rows).sort_values("Getiri %", ascending=False)
            # Özet istatistikler
            c1, c2, c3, c4 = st.columns(4)
            profitable = scan_df[scan_df["Getiri %"] > 0]
            c1.metric("Karlı Hisse", f"{len(profitable)}/{len(scan_df)}")
            c2.metric("Ort. Getiri", f"%{scan_df['Getiri %'].mean():.1f}")
            c3.metric("Ort. Win Rate", f"%{scan_df['Win Rate %'].mean():.1f}")
            c4.metric("Ort. Sharpe", f"{scan_df['Sharpe'].mean():.2f}")

            # Renk kodlu tablo
            def color_return(val):
                color = "#2ecc71" if val > 0 else "#e74c3c"
                return f"color: {color}; font-weight: bold"

            styled = scan_df.style.applymap(color_return, subset=["Getiri %"])
            st.dataframe(styled, use_container_width=True, height=600)
        else:
            st.warning("Hiçbir hisse için backtest sonucu alınamadı.")

# ============================================================
# TAB5 — RL AJANI
# ============================================================
with tab5:
    st.header("🤖 RL Ajanı Backtest")
    st.caption("Eğitilmiş PPO modelini seçili hisse üzerinde test eder ve kural tabanlı stratejiyle karşılaştırır.")

    import os
    rl_model_exists = os.path.exists("models/ppo_tradebot.zip") or os.path.exists("models/best_model.zip")

    if not rl_model_exists:
        st.warning("⚠️ Eğitilmiş model bulunamadı.")
        st.info(
            "Modeli eğitmek için terminalde şu komutu çalıştırın:\n\n"
            "```bash\npython rl_trainer.py --timesteps 1000000\n```\n\n"
            "Eğitim ~45-60 dakika sürer. Hızlı test için:\n\n"
            "```bash\npython rl_trainer.py --timesteps 200000\n```"
        )
        st.divider()
        st.subheader("📋 Eğitimde Kullanılan Parametreler")
        st.markdown("""
        | Kategori | Parametreler |
        |---|---|
        | **Trend** | Price/SMA50, Price/SMA200, SMA50/SMA200, SuperTrend yönü, ADX |
        | **Momentum** | RSI, MACD Hist/ATR, 1d/5d/20d getiri |
        | **Volatilite** | ATR/Fiyat, Bollinger genişliği |
        | **Hacim** | Hacim / 20g ort |
        | **Pozisyon** | Açık/Kapalı, Kâr/Zarar %, Tutulan gün, Drawdown |
        | **Piyasa** | BIST100 5 günlük getiri |

        **Algoritma:** PPO (Proximal Policy Optimization)
        **Ağ:** 256 → 256 → 128 nöron
        **Aksiyonlar:** 0=BEKLE · 1=AL · 2=SAT
        **Ödül:** Günlük getiri − işlem maliyeti − drawdown cezası
        """)
    else:
        rl_col1, rl_col2 = st.columns([2, 1])
        with rl_col1:
            rl_display = [s.replace(".IS", "") for s in BIST50_SYMBOLS]
            rl_sel = st.selectbox("Hisse", rl_display,
                                  index=rl_display.index("AKBNK"),
                                  key="rl_sym")
            rl_symbol = rl_sel + ".IS"
        with rl_col2:
            rl_period = st.selectbox("Periyot", ["1y", "2y", "3y"], index=1, key="rl_period")

        rl_compare = st.checkbox("Kural tabanlı stratejiyle karşılaştır (yavaş ~1-2 dk)", value=False, key="rl_compare")

        if st.button("▶️ RL Backtest Çalıştır", type="primary", key="rl_run"):
            from rl_backtester import run_rl_backtest

            with st.spinner(f"RL ajanı {rl_symbol} üzerinde test ediliyor..."):
                rl_res = run_rl_backtest(rl_symbol, period=rl_period)

            # Kural tabanlı backtest (opsiyonel)
            kb_res = None
            if rl_compare:
                with st.spinner(f"Kural tabanlı strateji {rl_symbol} üzerinde test ediliyor..."):
                    from backtester import run_backtest
                    kb_res = run_backtest(rl_symbol, period=rl_period)

            if rl_res.get("status") != "ok":
                st.error(f"Backtest başarısız: {rl_res.get('status')}")
            else:
                ret = rl_res["total_return_pct"]
                bh  = rl_res["buy_and_hold_pct"]

                # ── 3'lü karşılaştırma metrikleri ──────────────────────────
                if kb_res:
                    kb_ret = kb_res["total_return_pct"]
                    st.subheader("📊 Karşılaştırma")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🤖 RL Ajanı",        f"%{ret:+.1f}",
                               delta=f"{ret - bh:+.1f}% vs B&H")
                    c2.metric("📐 Kural Tabanlı",    f"%{kb_ret:+.1f}",
                               delta=f"{kb_ret - bh:+.1f}% vs B&H")
                    c3.metric("📈 Buy & Hold",       f"%{bh:+.1f}")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("RL Sharpe",           f"{rl_res['sharpe']:.2f}")
                    d2.metric("KB Sharpe",           f"{kb_res['sharpe']:.2f}")
                    d1.metric("RL Max Düşüş",        f"%{rl_res['max_drawdown']:.1f}")
                    d2.metric("KB Max Düşüş",        f"%{kb_res['max_drawdown_pct']:.1f}")
                    d1.metric("RL İşlem / Win",      f"{rl_res['n_trades']} / %{rl_res['win_rate']:.0f}")
                    d2.metric("KB İşlem / Win",      f"{kb_res['total_trades']} / %{kb_res['win_rate']:.0f}")
                else:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("RL Getiri", f"%{ret:+.1f}",
                               delta=f"{ret - bh:+.1f}% vs B&H")
                    m2.metric("Buy & Hold", f"%{bh:+.1f}")
                    m3.metric("İşlem Sayısı", rl_res["n_trades"])
                    m4.metric("Win Rate", f"%{rl_res['win_rate']:.0f}")
                    m5.metric("Max Drawdown", f"%{rl_res['max_drawdown']:.1f}")

                # ── Grafik ─────────────────────────────────────────────────
                curve  = rl_res.get("portfolio_curve")
                trades = rl_res.get("trades", [])
                initial_capital = 100_000.0

                if curve is not None and len(curve) > 0:
                    import plotly.graph_objects as go
                    fig_rl = go.Figure()

                    # RL portfolio çizgisi
                    fig_rl.add_trace(go.Scatter(
                        x=curve.index, y=curve.values,
                        name="RL Ajanı",
                        line=dict(color="#3498db", width=2),
                        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                    ))

                    # Kural tabanlı eğrisi
                    if kb_res and "df" in kb_res:
                        kb_eq = kb_res["df"]["Equity"].dropna()
                        # RL ile aynı başlangıç noktasına normalize et
                        if len(kb_eq) > 0:
                            kb_eq = kb_eq * (initial_capital / kb_eq.iloc[0])
                            fig_rl.add_trace(go.Scatter(
                                x=kb_eq.index, y=kb_eq.values,
                                name="Kural Tabanlı",
                                line=dict(color="#f39c12", width=2, dash="dot"),
                                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                            ))

                    # B&H referans çizgisi
                    if len(curve) > 0:
                        from rl_backtester import run_rl_backtest as _rl  # noqa
                        import yfinance as yf
                        try:
                            _bh_df = yf.Ticker(rl_symbol).history(period=rl_period, interval="1d")
                            if not _bh_df.empty:
                                _bh_close = _bh_df["Close"].loc[curve.index[0]:curve.index[-1]]
                                _bh_curve = initial_capital * _bh_close / _bh_close.iloc[0]
                                fig_rl.add_trace(go.Scatter(
                                    x=_bh_curve.index, y=_bh_curve.values,
                                    name="Buy & Hold",
                                    line=dict(color="#95a5a6", width=1, dash="dash"),
                                    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                                ))
                        except Exception:
                            pass

                    # AL işaretleri
                    buy_dates = [t["buy_date"] for t in trades if t.get("buy_date")]
                    buy_vals  = [curve.iloc[curve.index.get_indexer([d], method="nearest")[0]]
                                 for d in buy_dates]
                    if buy_dates:
                        fig_rl.add_trace(go.Scatter(
                            x=buy_dates, y=buy_vals, mode="markers", name="AL",
                            marker=dict(symbol="triangle-up", color="#2ecc71", size=14,
                                        line=dict(color="white", width=1)),
                            hovertemplate="AL<br>%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                        ))

                    # SAT işaretleri
                    sell_dates = [t["sell_date"] for t in trades if t.get("sell_date")]
                    sell_vals  = [curve.iloc[curve.index.get_indexer([d], method="nearest")[0]]
                                  for d in sell_dates]
                    if sell_dates:
                        fig_rl.add_trace(go.Scatter(
                            x=sell_dates, y=sell_vals, mode="markers", name="SAT",
                            marker=dict(symbol="triangle-down", color="#e74c3c", size=14,
                                        line=dict(color="white", width=1)),
                            hovertemplate="SAT<br>%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                        ))

                    fig_rl.update_layout(
                        title=f"Strateji Karşılaştırması — {rl_symbol}",
                        template="plotly_dark",
                        xaxis_title="Tarih",
                        yaxis_title="Portföy Değeri (TL)",
                        height=480,
                        legend=dict(orientation="h", y=1.05),
                    )
                    st.plotly_chart(fig_rl, use_container_width=True)

                # ── İşlem listesi ───────────────────────────────────────────
                if trades:
                    st.subheader(f"📋 RL İşlemleri ({len(trades)} adet)")
                    tr_df = pd.DataFrame(trades)
                    def _color_pnl(val):
                        return f"color: {'#2ecc71' if val > 0 else '#e74c3c'}; font-weight: bold"
                    styled_tr = tr_df.style.applymap(_color_pnl, subset=["pnl_pct"])
                    st.dataframe(styled_tr, use_container_width=True)
                else:
                    st.info("RL ajanı bu periyotta hiç işlem yapmadı.")

        # ── Tüm BIST50 RL Taraması ───────────────────────────────────────
        st.divider()
        st.subheader("🔍 Tüm BIST50'yi RL ile Tara")
        st.caption("RL ajanını tüm BIST50 hisselerinde çalıştırır ve kural tabanlı stratejiyle karşılaştırır.")

        rl_scan_period = st.selectbox("Tarama Periyodu", ["1y", "2y"], index=1, key="rl_scan_period")

        if st.button("🤖 Tüm BIST50'yi RL ile Tara", type="primary", key="rl_scan_all"):
            from rl_backtester import run_rl_backtest

            rl_rows = []
            rl_progress = st.progress(0, text="RL taraması başlıyor...")

            for i, sym in enumerate(BIST50_SYMBOLS):
                rl_progress.progress((i + 1) / len(BIST50_SYMBOLS),
                                     text=f"{sym} ({i+1}/{len(BIST50_SYMBOLS)})")
                res = run_rl_backtest(sym, period=rl_scan_period)
                if res.get("status") == "ok":
                    rl_rows.append({
                        "Hisse":          sym.replace(".IS", ""),
                        "RL Getiri (%)":  res["total_return_pct"],
                        "Al-Tut (%)":     res["buy_and_hold_pct"],
                        "RL Farkı (%)":   round(res["total_return_pct"] - res["buy_and_hold_pct"], 1),
                        "İşlem Sayısı":   res["n_trades"],
                        "Kazanma (%)":    res["win_rate"],
                        "Max Düşüş (%)":  res["max_drawdown"],
                        "Sharpe":         res["sharpe"],
                    })

            rl_progress.empty()

            if rl_rows:
                rl_scan_df = pd.DataFrame(rl_rows).sort_values("RL Getiri (%)", ascending=False)

                # Özet istatistikler
                sc1, sc2, sc3, sc4 = st.columns(4)
                rl_profitable = rl_scan_df[rl_scan_df["RL Getiri (%)"] > 0]
                rl_beat_bh    = rl_scan_df[rl_scan_df["RL Farkı (%)"] > 0]
                sc1.metric("Karlı Hisse",    f"{len(rl_profitable)}/{len(rl_scan_df)}")
                sc2.metric("B&H'ı Yenen",    f"{len(rl_beat_bh)}/{len(rl_scan_df)}")
                sc3.metric("Ort. RL Getiri", f"%{rl_scan_df['RL Getiri (%)'].mean():.1f}")
                sc4.metric("Ort. RL vs B&H", f"%{rl_scan_df['RL Farkı (%)'].mean():+.1f}")

                # Renk kodlu tablo
                def _color_rl(val):
                    return f"color: {'#2ecc71' if val > 0 else '#e74c3c'}; font-weight: bold"

                styled_rl = rl_scan_df.style.applymap(
                    _color_rl, subset=["RL Getiri (%)", "RL Farkı (%)"]
                )
                st.dataframe(styled_rl, use_container_width=True, height=600)
            else:
                st.warning("Hiçbir hisse için RL backtest sonucu alınamadı.")

# ============================================================
# TAB 6 — ROTASYON STRATEJİSİ
# ============================================================
with tab6:
    st.subheader("🔄 Rotasyon Stratejisi Backtesti")
    st.caption("En güçlü trende girer, kârını alıp başka fırsata geçer. Tüm BIST50 taranır.")

    with st.expander("⚙️ Ayarlar", expanded=True):
        rc1, rc2, rc3, rc4 = st.columns(4)
        rot_period     = rc1.selectbox("Periyot", ["1y", "2y", "3y"], index=0, key="rot_period")
        rot_capital    = rc2.number_input("Başlangıç Sermaye", value=100000, step=10000, key="rot_capital")
        rot_entry_thr  = rc3.slider("Giriş Skoru Eşiği", 55, 80, 62, key="rot_entry")
        rot_trail      = rc4.slider("Trailing Stop (%)", 5, 15, 8, key="rot_trail")
        rc5, rc6 = st.columns(2)
        rot_min_hold   = rc5.slider("Min. Tutma (gün)", 3, 20, 5, key="rot_min_hold")
        rot_exit_score = rc6.slider("Çıkış Skoru Eşiği", 35, 60, 45, key="rot_exit_score")

    if st.button("▶️ Rotasyon Backtestini Çalıştır", key="run_rotation"):
        from rotator import run_rotation_backtest
        import yfinance as yf

        progress_bar = st.progress(0, text="Başlatılıyor...")

        def _prog(pct, msg):
            progress_bar.progress(min(pct, 1.0), text=msg)

        result = run_rotation_backtest(
            symbols=BIST50_SYMBOLS,
            period=rot_period,
            initial_capital=float(rot_capital),
            entry_threshold=float(rot_entry_thr),
            trail_pct=float(rot_trail),
            min_hold_days=int(rot_min_hold),
            exit_score_threshold=float(rot_exit_score),
            progress_callback=_prog,
        )
        progress_bar.empty()

        if result is None:
            st.error("Yeterli veri bulunamadı.")
        else:
            st.session_state["rot_result"] = result
            st.session_state["rot_period_used"] = rot_period
            st.session_state["rot_capital_used"] = rot_capital

    if "rot_result" in st.session_state:
        import yfinance as yf
        result     = st.session_state["rot_result"]
        rot_period = st.session_state.get("rot_period_used", rot_period)
        rot_capital = st.session_state.get("rot_capital_used", rot_capital)
        if True:
            m = result["metrics"]

            # ── Metrik kartları ─────────────────────────────────────────
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            delta_color = "normal" if m["Toplam Getiri (%)"] >= 0 else "inverse"
            mc1.metric("Toplam Getiri", f"%{m['Toplam Getiri (%)']:+.1f}")
            mc2.metric("Sharpe",        f"{m['Sharpe']:.2f}")
            mc3.metric("Max Düşüş",     f"%{m['Max Düşüş (%)']:.1f}")
            mc4.metric("İşlem Sayısı",  m["İşlem Sayısı"])
            mc5.metric("Kazanma Oranı", f"%{m['Kazanma (%)']:.1f}")

            # ── Portföy eğrisi ──────────────────────────────────────────
            import plotly.graph_objects as go
            curve = result["portfolio_curve"]

            # B&H karşılaştırması: XU100.IS endeksi
            try:
                bh_df = yf.Ticker("XU100.IS").history(period=rot_period, interval="1d")
                bh_norm = bh_df["Close"] / bh_df["Close"].iloc[0] * float(rot_capital)
                bh_norm.index = bh_norm.index.normalize()
            except Exception:
                bh_norm = None

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve.index, y=curve.values,
                name="Rotasyon Stratejisi",
                line=dict(color="#2ecc71", width=2),
            ))
            if bh_norm is not None:
                fig.add_trace(go.Scatter(
                    x=bh_norm.index, y=bh_norm.values,
                    name="XU100 B&H",
                    line=dict(color="#888", width=1.5, dash="dash"),
                ))

            # Giriş/çıkış noktalarını işaretle
            trades_df = pd.DataFrame(result["trades"])
            if not trades_df.empty:
                # Timezone uyumsuzluğunu gider — ikisini de tz-naive yap
                curve_naive = curve.copy()
                curve_naive.index = curve_naive.index.tz_localize(None) if curve_naive.index.tz is None else curve_naive.index.tz_convert(None)
                entry_dates = pd.to_datetime(trades_df["Giriş Tarihi"]).dt.tz_localize(None)
                entry_vals  = curve_naive.reindex(entry_dates.values, method="nearest")
                exit_dates  = pd.to_datetime(trades_df["Çıkış Tarihi"]).dt.tz_localize(None)
                exit_vals   = curve_naive.reindex(exit_dates.values, method="nearest")

                fig.add_trace(go.Scatter(
                    x=entry_dates, y=entry_vals.values,
                    mode="markers", name="Giriş",
                    marker=dict(color="#3498db", size=8, symbol="triangle-up"),
                ))
                fig.add_trace(go.Scatter(
                    x=exit_dates, y=exit_vals.values,
                    mode="markers", name="Çıkış",
                    marker=dict(color="#e74c3c", size=8, symbol="triangle-down"),
                ))

            fig.update_layout(
                title="Portföy Değeri",
                xaxis_title="Tarih",
                yaxis_title="Portföy (TL)",
                template="plotly_dark",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── İşlem geçmişi ────────────────────────────────────────────
            if not trades_df.empty:
                def _color_trade(val):
                    if isinstance(val, (int, float)):
                        return f"color: {'#2ecc71' if val > 0 else '#e74c3c'}; font-weight: bold"
                    return ""

                st.subheader(f"📋 İşlem Geçmişi ({len(trades_df)} işlem)")
                styled_trades = trades_df.style.applymap(
                    _color_trade, subset=["Kâr/Zarar", "Getiri %"]
                )
                st.dataframe(styled_trades, use_container_width=True, height=300)

                # ── Hisse Detay Grafikleri ────────────────────────────────
                st.subheader("📈 Hisse Detayları — Al/Sat Noktaları")
                unique_syms = trades_df["Hisse"].unique().tolist()

                # Butonlar
                btn_cols = st.columns(min(len(unique_syms), 8))
                if "rot_selected_sym" not in st.session_state or st.session_state["rot_selected_sym"] not in unique_syms:
                    st.session_state["rot_selected_sym"] = unique_syms[0]

                for bi, sym_name in enumerate(unique_syms):
                    col_i = bi % len(btn_cols)
                    sym_trades = trades_df[trades_df["Hisse"] == sym_name]
                    total_pct  = sym_trades["Getiri %"].sum()
                    label = f"{'🟢' if total_pct >= 0 else '🔴'} {sym_name}"
                    if btn_cols[col_i].button(label, key=f"rot_btn_{sym_name}"):
                        st.session_state["rot_selected_sym"] = sym_name

                # Seçili hisse grafiği
                sel_sym  = st.session_state["rot_selected_sym"]
                sel_full = sel_sym + ".IS"
                sym_trades_sel = trades_df[trades_df["Hisse"] == sel_sym]

                try:
                    price_df = yf.Ticker(sel_full).history(period=rot_period, interval="1d")
                    price_df.index = price_df.index.tz_convert(None) if price_df.index.tz else price_df.index

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=price_df.index, y=price_df["Close"],
                        name=sel_sym, line=dict(color="#aaaaaa", width=1.5),
                    ))

                    for _, tr in sym_trades_sel.iterrows():
                        buy_dt  = pd.Timestamp(tr["Giriş Tarihi"])
                        sell_dt = pd.Timestamp(tr["Çıkış Tarihi"])
                        color   = "#2ecc71" if tr["Getiri %"] >= 0 else "#e74c3c"

                        # Al noktası
                        fig2.add_trace(go.Scatter(
                            x=[buy_dt], y=[tr["Giriş Fiyatı"]],
                            mode="markers+text",
                            marker=dict(color="#3498db", size=12, symbol="triangle-up"),
                            text=[f"AL<br>{tr['Giriş Fiyatı']:.2f}"],
                            textposition="top center",
                            textfont=dict(size=10, color="#3498db"),
                            showlegend=False,
                        ))
                        # Sat noktası
                        fig2.add_trace(go.Scatter(
                            x=[sell_dt], y=[tr["Çıkış Fiyatı"]],
                            mode="markers+text",
                            marker=dict(color=color, size=12, symbol="triangle-down"),
                            text=[f"SAT<br>{tr['Çıkış Fiyatı']:.2f}<br>{tr['Getiri %']:+.1f}%"],
                            textposition="bottom center",
                            textfont=dict(size=10, color=color),
                            showlegend=False,
                        ))
                        # Al-Sat arası bölge
                        fig2.add_vrect(
                            x0=buy_dt, x1=sell_dt,
                            fillcolor=color, opacity=0.07,
                            layer="below", line_width=0,
                        )

                    total_kar = sym_trades_sel["Kâr/Zarar"].sum()
                    fig2.update_layout(
                        title=f"{sel_sym} — {len(sym_trades_sel)} işlem | Toplam: {total_kar:+.0f} TL",
                        xaxis_title="Tarih",
                        yaxis_title="Fiyat",
                        template="plotly_dark",
                        height=450,
                        showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # Bu hissenin işlem detayları
                    st.dataframe(
                        sym_trades_sel.style.applymap(_color_trade, subset=["Kâr/Zarar", "Getiri %"]),
                        use_container_width=True, height=200,
                    )
                except Exception as e:
                    st.warning(f"{sel_sym} grafiği yüklenemedi: {e}")

                # En çok girilen hisseler
                st.subheader("🏆 Özet")
                top_syms = trades_df.groupby("Hisse").agg(
                    İşlem=("Hisse", "count"),
                    Ort_Getiri=("Getiri %", "mean"),
                    Toplam_Kar=("Kâr/Zarar", "sum"),
                ).sort_values("Toplam_Kar", ascending=False).reset_index()
                top_syms.columns = ["Hisse", "İşlem Sayısı", "Ort. Getiri %", "Toplam Kâr (TL)"]
                top_syms["Ort. Getiri %"] = top_syms["Ort. Getiri %"].round(2)
                top_syms["Toplam Kâr (TL)"] = top_syms["Toplam Kâr (TL)"].round(2)
                st.dataframe(top_syms.style.applymap(
                    _color_trade, subset=["Ort. Getiri %", "Toplam Kâr (TL)"]
                ), use_container_width=True)

# ============================================================
# TAB 7 — TEMEL ANALİZ (Piotroski F-Score)
# ============================================================
with tab7:
    from fundamental import get_piotroski

    st.subheader("📋 Temel Analiz — Piotroski F-Score")
    st.caption("9 objektif finansal kriter üzerinden şirket sağlığını ölçer. Veri: yfinance (yıllık finansallar).")

    fa_display_names = [s.replace(".IS", "") for s in BIST50_SYMBOLS]
    fa_selected = st.selectbox("Hisse Seçin", fa_display_names,
                               index=fa_display_names.index("THYAO"), key="fa_symbol")
    fa_symbol = fa_selected + ".IS"

    if st.button("📊 F-Score Hesapla", key="fa_run"):
        with st.spinner(f"{fa_selected} temel verileri çekiliyor..."):
            fa_result = get_piotroski(fa_symbol)
        st.session_state["fa_result"] = fa_result
        st.session_state["fa_symbol_used"] = fa_selected

    if "fa_result" in st.session_state:
        fa_result = st.session_state["fa_result"]
        fa_sym    = st.session_state.get("fa_symbol_used", fa_selected)

        if fa_result["error"]:
            st.error(f"Veri alınamadı: {fa_result['error']}")
        else:
            score = fa_result["score"]
            label = fa_result["label"]

            # ── Skor göstergesi ───────────────────────────────────────
            score_colors = {"Zayıf": "#d32f2f", "Nötr": "#f57c00", "İyi": "#388e3c", "Güçlü": "#1565c0"}
            score_color  = score_colors.get(label, "#555")

            st.markdown(f"""
            <div style="background:{score_color};border-radius:12px;padding:20px 30px;display:inline-block;margin-bottom:16px;">
                <span style="font-size:2.5rem;font-weight:bold;color:#fff;">{score}/9</span>
                <span style="font-size:1.2rem;color:#fff;margin-left:12px;">{label}</span>
                <br><span style="color:#eee;font-size:0.85rem;">{fa_sym} Piotroski F-Score</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Kriter tablosu ────────────────────────────────────────
            st.markdown("### Kriter Detayı")
            groups = ["Karlılık", "Kaldıraç & Likidite", "Operasyonel Verimlilik"]
            group_icons = {"Karlılık": "💰", "Kaldıraç & Likidite": "🏦", "Operasyonel Verimlilik": "⚙️"}

            for grp in groups:
                grp_criteria = [c for c in fa_result["criteria"] if c["group"] == grp]
                grp_score    = sum(1 for c in grp_criteria if c["passed"])
                grp_total    = len(grp_criteria)
                st.markdown(f"**{group_icons[grp]} {grp}** — {grp_score}/{grp_total}")

                for c in grp_criteria:
                    icon = "✅" if c["passed"] else "❌"
                    st.markdown(
                        f"{icon} **{c['name']}** &nbsp;&nbsp; `{c['value']}` "
                        f"<span style='color:#888;font-size:0.8em'>— {c['desc']}</span>",
                        unsafe_allow_html=True,
                    )
                st.markdown("")

            # ── Metrik grupları ───────────────────────────────────────
            st.markdown("---")
            metrics = fa_result.get("metrics", {})
            group_icons = {
                "Değerleme":       "💰",
                "Kârlılık":        "📈",
                "Finansal Sağlık": "🏦",
                "Büyüme (YoY)":    "🚀",
            }
            for grp_name, grp_items in metrics.items():
                icon = group_icons.get(grp_name, "📊")
                st.markdown(f"### {icon} {grp_name}")
                cols = st.columns(min(len(grp_items), 4))
                for i, (label, value, desc) in enumerate(grp_items):
                    display = str(value) if value is not None else "—"
                    cols[i % 4].metric(label, display, help=desc)
                st.markdown("")

            # ── Yorumlama kılavuzu ────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            **F-Score yorumlama:**
            | Skor | Anlam | Yaklaşım |
            |------|-------|----------|
            | 8–9  | Güçlü finansal yapı | Teknik sinyal + F-Score birlikte değerlendir |
            | 6–7  | İyi | Pozitif sinyal |
            | 3–5  | Nötr | Temkinli yaklaş |
            | 0–2  | Zayıf finansal yapı | Kaçın veya short fırsatı |
            """)

    # ── BIST50 Tarama ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 BIST50 F-Score Tarama")
    st.caption("Tüm BIST50 hisselerini Piotroski skoru ve temel metriklerle sıralar. "
               "İlk tarama ~2 dk sürebilir; sonraki taramalar önbellekten hızlıdır.")

    if st.button("🔍 Tüm BIST50'yi Tara", key="fa_scan_all"):
        def _mv(m, group, label):
            """Metrics dict'inden değer al; None ise '—' döndür."""
            for lbl, val, _ in m.get(group, []):
                if lbl == label:
                    return val if val is not None else "—"
            return "—"

        def _mvn(m, group, label):
            """Numeric metrikler için None-safe: None → None (DataFrame'de boş)."""
            for lbl, val, _ in m.get(group, []):
                if lbl == label:
                    return val  # float veya None
            return None

        rows = []
        prog_bar  = st.progress(0)
        status_tx = st.empty()
        total = len(BIST50_SYMBOLS)

        for i, sym in enumerate(BIST50_SYMBOLS):
            short = sym.replace(".IS", "")
            status_tx.text(f"Taranıyor: {short} ({i + 1}/{total})")
            try:
                res = get_piotroski(sym)
            except Exception as ex:
                res = {"score": None, "label": "Hata", "metrics": {}, "error": str(ex)}
            prog_bar.progress((i + 1) / total)

            if res.get("error") or res["score"] is None:
                rows.append({
                    "Hisse": short, "F-Score": None, "Durum": res.get("label", "Hata"),
                    "F/K": None, "PD/DD": None, "EV/FAVÖK": None,
                    "Net Marj": "—", "ROE": "—", "Gelir Büy.": "—", "Temettü": "—",
                })
                continue

            m = res["metrics"]
            rows.append({
                "Hisse":      short,
                "F-Score":    res["score"],
                "Durum":      res["label"],
                "F/K":        _mvn(m, "Değerleme", "F/K"),
                "PD/DD":      _mvn(m, "Değerleme", "PD/DD"),
                "EV/FAVÖK":   _mvn(m, "Değerleme", "EV/FAVÖK"),
                "Net Marj":   _mv(m, "Kârlılık", "Net Marj"),
                "ROE":        _mv(m, "Kârlılık", "ROE"),
                "Gelir Büy.": _mv(m, "Büyüme (YoY)", "Gelir Büyümesi"),
                "Temettü":    _mv(m, "Büyüme (YoY)", "Temettü Verimi"),
            })

        prog_bar.empty()
        status_tx.empty()
        ok_count = sum(1 for r in rows if r["F-Score"] is not None)
        st.success(f"✅ {ok_count}/{total} hisse başarıyla tarandı.")
        st.session_state["fa_scan_result"] = rows

    if "fa_scan_result" in st.session_state:
        scan_rows = st.session_state["fa_scan_result"]
        scan_df   = pd.DataFrame(scan_rows)

        # Filtre
        filter_opts   = ["Tümü", "Güçlü", "İyi", "Nötr", "Zayıf", "Hata"]
        scan_filter   = st.radio("Filtre", filter_opts, horizontal=True, key="fa_scan_filter")
        if scan_filter != "Tümü":
            scan_df = scan_df[scan_df["Durum"] == scan_filter]

        scan_df = (scan_df
                   .sort_values("F-Score", ascending=False, na_position="last")
                   .reset_index(drop=True))
        scan_df.index = scan_df.index + 1  # 1-tabanlı sıra

        def _color_fscore(val):
            if not isinstance(val, (int, float)):
                return "color: #888"
            if val >= 8:
                return "background-color: #1b5e20; color: white"
            if val >= 6:
                return "background-color: #388e3c; color: white"
            if val >= 3:
                return "background-color: #f57c00; color: white"
            return "background-color: #c62828; color: white"

        styled = scan_df.style.applymap(_color_fscore, subset=["F-Score"])
        st.dataframe(styled, use_container_width=True, height=600)

# ============================================================
# TAB 8 — KOMBİNE STRATEJİ (Rotasyon + RL + Filtreler)
# ============================================================
with tab8:
    from combo_backtester import run_combo_backtest

    st.subheader("🧬 Kombine Strateji — Rotasyon + RL + Temel Filtreler")
    st.caption(
        "Rotasyon momentum skoru ile hisse seçer, "
        "MA200 / F-Score filtrelerinden geçirip pozisyona girer. "
        "RL modeli pozisyon içindeyken erken çıkış sinyali üretir."
    )

    # ── Ayarlar ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Strateji Ayarları", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            combo_date_mode = st.radio("Tarih Modu", ["Periyot", "Sabit Aralık"],
                                       horizontal=True, key="cb_date_mode")
            if combo_date_mode == "Periyot":
                combo_period = st.selectbox("Periyot", ["6mo", "1y", "2y", "3y"],
                                            index=2, key="cb_period")
                combo_start = combo_end = None
            else:
                import datetime
                combo_start = st.date_input("Başlangıç", value=datetime.date(2024, 1, 1), key="cb_start")
                combo_end   = st.date_input("Bitiş",     value=datetime.date(2025, 12, 31), key="cb_end")
                combo_period = None
            combo_capital = st.number_input("Başlangıç Sermaye (TL)",
                                            value=100_000, step=10_000, key="cb_capital")
        with c2:
            combo_entry   = st.slider("Giriş Skoru Eşiği", 50, 85, 65, key="cb_entry")
            combo_trail   = st.slider("Trailing Stop (%)", 3.0, 15.0, 6.0, 0.5, key="cb_trail")
            combo_minhold = st.slider("Min. Tutma (gün)", 1, 20, 7, key="cb_minhold")
            combo_exit_sc = st.slider("Çıkış Skoru Eşiği", 30, 60, 48, key="cb_exit_sc",
                                      help="Skor bu değerin altına düşerse çık")
        with c3:
            combo_use_ma200    = st.toggle("📈 MA200 Filtresi", value=True, key="cb_ma200")
            combo_min_fscore   = st.slider("🏦 Min F-Score", 0, 9, 5, key="cb_fscore",
                                           help="0 = filtre kapalı")
            combo_use_sector   = st.toggle("🏭 Sektör Endeksi Filtresi", value=False, key="cb_sektor",
                                           help="XBANK/XELKT/... MA50 altındaysa o sektöre girme")
            combo_use_rl       = st.toggle("🤖 RL Çıkış Sinyali", value=True, key="cb_rl",
                                           help="PPO modeli SAT diyorsa pozisyonu erken kapat")

    if st.button("▶ Kombine Backtest Çalıştır", key="cb_run", type="primary"):
        prog_ph   = st.progress(0)
        status_ph = st.empty()

        def _cb_prog(pct, msg):
            prog_ph.progress(pct)
            status_ph.text(msg)

        with st.spinner("Hesaplanıyor..."):
            cb_result = run_combo_backtest(
                symbols               = BIST50_SYMBOLS,
                period                = combo_period,
                start_date            = str(combo_start) if combo_start else None,
                end_date              = str(combo_end)   if combo_end   else None,
                initial_capital       = float(combo_capital),
                entry_threshold       = combo_entry,
                trail_pct             = combo_trail,
                min_hold_days         = combo_minhold,
                exit_score_threshold  = combo_exit_sc,
                use_ma200             = combo_use_ma200,
                min_fscore            = combo_min_fscore,
                use_sector_filter     = combo_use_sector,
                use_rl                = combo_use_rl,
                progress_callback     = _cb_prog,
            )
        prog_ph.empty(); status_ph.empty()

        if "error" in cb_result:
            st.error(cb_result["error"])
        else:
            st.session_state["cb_result"] = cb_result

    if "cb_result" in st.session_state:
        cb = st.session_state["cb_result"]
        m  = cb["metrics"]

        # ── Metrik kartları ────────────────────────────────────────────
        st.markdown("---")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        ret_color = "normal" if m["Toplam Getiri (%)"] >= 0 else "inverse"
        mc1.metric("Toplam Getiri",  f"%{m['Toplam Getiri (%)']:+.1f}")
        mc2.metric("Sharpe",          f"{m['Sharpe']:.2f}")
        mc3.metric("Max Düşüş",       f"%{m['Max Düşüş (%)']:.1f}")
        mc4.metric("İşlem Sayısı",    m["İşlem Sayısı"])
        mc5.metric("Kazanma Oranı",   f"%{m['Kazanma (%)']:.1f}")

        # Aktif filtreler
        fu = cb.get("filters_used", {})
        badges = []
        if fu.get("ma200"):   badges.append("✅ MA200")
        else:                 badges.append("⬜ MA200")
        if fu.get("fscore"):  badges.append(f"✅ F-Score≥{combo_min_fscore}")
        else:                 badges.append("⬜ F-Score")
        if fu.get("sektor"):  badges.append("✅ Sektör")
        else:                 badges.append("⬜ Sektör")
        if fu.get("rl"):      badges.append("✅ RL Çıkış")
        else:                 badges.append("⬜ RL Çıkış")
        st.caption("Aktif filtreler: " + " &nbsp; ".join(badges))

        # ── Portföy grafiği ───────────────────────────────────────────
        st.markdown("#### Portföy Değeri")
        curve = cb["portfolio_curve"]

        try:
            import plotly.graph_objects as go
            if combo_start and combo_end:
                bist_bh = yf.Ticker("XU100.IS").history(start=str(combo_start), end=str(combo_end))["Close"]
            else:
                bist_bh = yf.Ticker("XU100.IS").history(period=combo_period)["Close"]
            bist_bh = (bist_bh / bist_bh.iloc[0] * float(combo_capital))
            bist_bh.index = bist_bh.index.tz_convert(None)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.index, y=curve.values,
                                     name="Kombine Strateji", line=dict(color="#00e676", width=2)))
            fig.add_trace(go.Scatter(x=bist_bh.index, y=bist_bh.values,
                                     name="XU100 B&H", line=dict(color="#aaa", width=1.5, dash="dash")))

            # Giriş/çıkış işaretleri
            trades_df = pd.DataFrame(cb["trades"]) if cb["trades"] else pd.DataFrame()
            if not trades_df.empty:
                entries = trades_df.copy()
                entries["x"] = pd.to_datetime(entries["Giriş Tarihi"])
                entries["y"] = entries["Giriş Fiyatı"]
                fig.add_trace(go.Scatter(
                    x=entries["x"], y=curve.reindex(entries["x"], method="nearest").values,
                    mode="markers", name="Giriş",
                    marker=dict(symbol="triangle-up", size=10, color="#42a5f5")
                ))
                exits = trades_df.copy()
                exits["x"] = pd.to_datetime(exits["Çıkış Tarihi"])
                fig.add_trace(go.Scatter(
                    x=exits["x"], y=curve.reindex(exits["x"], method="nearest").values,
                    mode="markers", name="Çıkış",
                    marker=dict(symbol="triangle-down", size=10, color="#ef5350")
                ))

            fig.update_layout(
                height=400, template="plotly_dark",
                xaxis_title="Tarih", yaxis_title="Portföy (TL)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.line_chart(curve)

        # ── İşlem geçmişi ─────────────────────────────────────────────
        if cb["trades"]:
            st.markdown(f"#### 📋 İşlem Geçmişi ({len(cb['trades'])} işlem)")
            trades_df = pd.DataFrame(cb["trades"])

            def _color_trade_cb(val):
                if isinstance(val, (int, float)):
                    return "color: #4caf50" if val > 0 else ("color: #f44336" if val < 0 else "")
                return ""

            st.dataframe(
                trades_df.style.applymap(_color_trade_cb, subset=["Kâr/Zarar", "Getiri %"]),
                use_container_width=True,
            )

        # ── Filtre günlüğü ─────────────────────────────────────────────
        if not cb["filter_log"].empty:
            with st.expander(f"🔍 Filtre Günlüğü ({len(cb['filter_log'])} kayıt)"):
                st.dataframe(cb["filter_log"], use_container_width=True, height=300)
