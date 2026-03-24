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
    
    # Elite Plus Metrikleri
    st.markdown("---")
    st.markdown("#### 🌟 Elite Plus & AI Hybrid")
    g1, g2 = st.columns(2)
    with g1:
        m_score = result.get("minervini_score", 0)
        st.metric("🏁 Minervini Trend", "ONAYLANDI" if m_score >= 7 else "Zayif", delta=f"{int(m_score)}/8")
    with g2:
        st.metric("📊 Z-Score (Risk)", f"{result.get('z_score', 0):.2f}")
        st.caption("İstatistiki Sapma")
        
    # Strateji Optimizasyonu
    st.markdown("---")
    st.markdown("#### 🎯 Strateji Optimizasyonu")
    c_ind = result.get('champion_indicator', 'Nötr')
    st.metric("Şampiyon İndikatör", c_ind)
    st.caption(f"Bu hissede tarihsel olarak en iyi çalışan indikatör.")

    # Faz 27: Haberler ve Katalizörler
    st.markdown("---")
    st.markdown("#### 📡 Haber Akışı & Piyasa Beklentileri (Faz 27)")
    n1, n2 = st.columns([2, 1])
    
    with n1:
        st.subheader("📰 Son Haberler (Investing)")
        news_items = result.get("news_items", [])
        if news_items:
            for item in news_items:
                sentiment_emoji = "✅" if item["sentiment"] == "Positive" else "❌" if item["sentiment"] == "Negative" else "⚪"
                st.markdown(f"{sentiment_emoji} [{item['title']}]({item['link']})")
        else:
            st.info("Bu hisse için güncel haber bulunamadı.")
            
    with n2:
        st.subheader("🌍 Analiz & Takvim")
        
        cats = result.get("catalysts", {})
        if cats:
            st.write(f"📅 **Bilanço:** {cats.get('next_earnings', 'Bilinmiyor')}")
            if cats.get("days_to_earnings") is not None:
                st.caption(f"({cats['days_to_earnings']} gün kaldı)")
            st.write(f"💰 **Temettü:** {cats.get('ex_dividend_date', 'Bilinmiyor')}")
            st.write(f"📈 **Verim:** %{cats.get('dividend_yield', 0)}")
        else:
            st.write("Katalizör verisi alınamadı.")

    # Faz 28: Mevsimsel Performans
    st.markdown("---")
    st.markdown("#### 🗓️ Mevsimsel Performans Karnesi (Faz 28)")
    
    seas_stats = result.get("seasonal_stats", [])
    if seas_stats:
        s1, s2 = st.columns([1, 1])
        
        # DataFrame hazırlığı (Chart için)
        seas_df = pd.DataFrame(seas_stats)
        
        with s1:
            st.subheader("📊 Aylık Başarı Oranı (%)")
            # Bar chart ile gösterim
            st.bar_chart(data=seas_df, x="month_name", y="win_rate", color="#2ecc71")
            st.caption("Bu grafikte her ayın tarihsel olarak kaç kez pozitif kapattığı (%) gösterilir.")

        with s2:
            st.subheader("📈 Ortalama Aylık Getiri (%)")
            st.bar_chart(data=seas_df, x="month_name", y="avg_return", color="#3498db")
            st.caption("Bu grafikte her ayın tarihsel ortalama getiri yüzdesi gösterilir.")
            
        # Önemli Ay Uyarısı
        curr_m = result.get("current_month", 0)
        curr_stat = next((item for item in seas_stats if item["month"] == curr_m), None)
        if curr_stat:
            st.warning(f"🔔 **Mevsimsel Bilgi:** Bu ay ({curr_stat['month_name']}) hisse tarihsel olarak %{curr_stat['win_rate']} başarı oranına ve %{curr_stat['avg_return']} ortalama getiriye sahip.")
    else:
        st.info("Mevsimsel veri oluşturulamadı (Yetersiz geçmiş veri).")

    # Faz 29: Elite Vision
    st.markdown("---")
    st.markdown("#### 🛡️ Elite Vision: Teknik Onaylar (Faz 29)")
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("🛡️ SuperTrend & Z-Score")
        st_dir = result.get("supertrend_dir", 0)
        st_status = "🟢 YÜKSELİŞ" if st_dir == 1 else "🔴 DÜŞÜŞ" if st_dir == -1 else "⚪ NÖTR"
        st.info(f"**SuperTrend Durumu:** {st_status}")
        
        z = result.get("z_score", 0)
        z_status = "🚨 AŞIRI ŞİŞMİŞ" if z > 2.5 else "🎁 AŞIRI UCUZ" if z < -2.5 else "✅ DENGELİ"
        st.write(f"**Z-Score Değeri:** {z:.2f} ({z_status})")
        
    with v2:
        st.subheader("💎 GMMA Trend Gücü")
        st.write(f"**Trend Ayrışması:** %{result.get('gmma_div', 0)}")
        st.write(f"**Grup Yayılımı:** %{result.get('gmma_spread', 0)}")
        if result.get("gmma_div", 0) > 2:
            st.success("Güçlü Kurumsal Alım Sinyali!")
        elif result.get("gmma_div", 0) < -2:
            st.error("Güçlü Satış Baskısı!")
        
    # Faz 25 Risk Metrikleri
    st.markdown("---")
    st.markdown("#### 🛡️ Risk & Tahmin Güvenliği (Faz 25)")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("🎲 MC Başarı Olasılığı", f"%{result.get('sim_success_prob')}")
    with r2:
        st.metric("🌊 Max Beklenen Zarar (VaR)", f"%{result.get('sim_risk_pct')}")
    with r3:
        st.metric("🛣️ Trend Otoyolu (RWB)", "AKTİF" if result.get("rwb_highway") else "Pasif", delta=f"Skor: {result.get('rwb_score')}/6")
    with r4:
        # Basit Kelly Hesabı
        k_score = portfolio_manager.calculate_kelly_criterion(result.get("sim_success_prob", 50)/100, 1.5)
        st.metric("🐢 Kelly Lot Oranı", f"%{k_score*100:.1f}")
        st.caption("Sermaye Kullanım Önerisi")

    # --- Phase 32: ELITE ULTRA LIVE EXIT ADVISOR ---
    st.markdown("---")
    st.markdown("#### 🛡️ Elite Ultra: Live Exit Advisor (Kâr Koruma)")
    e1, e2 = st.columns(2)
    with e1:
        st.subheader("📏 Önerilen Dinamik Stop")
        atr_stop = result.get("atr_trailing_stop", 0)
        current_price = result.get("price", 0)
        if atr_stop > 0:
            diff_pct = ((atr_stop - current_price) / current_price) * 100
            st.metric("İz Süren Stop Fiyatı", f"{atr_stop} ₺", delta=f"{diff_pct:.1f}%")
            st.caption("Bu seviyenin altında mum kapanışı stop sinyalidir.")
        else:
            st.info("Hesaplanıyor...")
            
    with e2:
        st.subheader("🔥 Hacimli Zirve (Blow-off) Riski")
        v_risk = result.get("volume_peak_risk", 0)
        if v_risk >= 90:
            st.error(f"KRİTİK RİSK: %{v_risk}")
            st.markdown("⚠️ **Hacim patlaması ve zirve fiyat!** Kârı kilitlemek için stopu anında %1'e çekmelisin.")
        elif v_risk >= 65:
            st.warning(f"YÜKSEK RİSK: %{v_risk}")
            st.markdown("🔔 **Dağıtım başlıyor olabilir.** Stopu çok daraltma zamanı.")
        elif v_risk >= 40:
            st.info(f"ORTA RİSK: %{v_risk}")
            st.markdown("💡 Hacim artışına dikkat edilmeli.")
        else:
            st.success("DÜŞÜK RİSK")
            st.markdown("✅ Olağandışı bir hacim/zirve şişmesi görülmedi.")

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Tekli Hisse Analizi", "🚀 BIST50 Tam Tarama", "🗺️ Piyasa Haritası", "🎲 Risk & Simülasyon Merkezi", "🤖 RL Ajanı"])

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

        if st.button("▶️ RL Backtest Çalıştır", type="primary", key="rl_run"):
            from rl_backtester import run_rl_backtest

            with st.spinner(f"RL ajanı {rl_symbol} üzerinde test ediliyor..."):
                rl_res = run_rl_backtest(rl_symbol, period=rl_period)

            if rl_res.get("status") != "ok":
                st.error(f"Backtest başarısız: {rl_res.get('status')}")
            else:
                # ── Özet metrikler ─────────────────────────────────────────
                m1, m2, m3, m4, m5 = st.columns(5)
                ret = rl_res["total_return_pct"]
                bh  = rl_res["buy_and_hold_pct"]
                m1.metric("RL Getiri", f"%{ret:+.1f}",
                           delta=f"{ret - bh:+.1f}% vs B&H")
                m2.metric("Buy & Hold", f"%{bh:+.1f}")
                m3.metric("İşlem Sayısı", rl_res["n_trades"])
                m4.metric("Win Rate", f"%{rl_res['win_rate']:.0f}")
                m5.metric("Max Drawdown", f"%{rl_res['max_drawdown']:.1f}")

                # ── Portfolio curve + Al/Sat işaretleri ────────────────────
                curve  = rl_res.get("portfolio_curve")
                trades = rl_res.get("trades", [])
                if curve is not None and len(curve) > 0:
                    import plotly.graph_objects as go
                    fig_rl = go.Figure()

                    # Ana portfolio çizgisi
                    fig_rl.add_trace(go.Scatter(
                        x=curve.index, y=curve.values,
                        name="RL Portfolio",
                        line=dict(color="#3498db", width=2),
                        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                    ))

                    # AL işaretleri (yeşil üçgen ▲)
                    buy_dates = [t["buy_date"]  for t in trades if t.get("buy_date")]
                    buy_vals  = [curve.asof(d)  for d in buy_dates if d in curve.index or True]
                    buy_vals  = [curve.iloc[curve.index.get_indexer([d], method="nearest")[0]]
                                 for d in buy_dates]
                    if buy_dates:
                        fig_rl.add_trace(go.Scatter(
                            x=buy_dates, y=buy_vals,
                            mode="markers",
                            name="AL",
                            marker=dict(symbol="triangle-up", color="#2ecc71", size=14,
                                        line=dict(color="white", width=1)),
                            hovertemplate="AL<br>%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                        ))

                    # SAT işaretleri (kırmızı üçgen ▼)
                    sell_dates = [t["sell_date"] for t in trades if t.get("sell_date")]
                    sell_vals  = [curve.iloc[curve.index.get_indexer([d], method="nearest")[0]]
                                  for d in sell_dates]
                    if sell_dates:
                        fig_rl.add_trace(go.Scatter(
                            x=sell_dates, y=sell_vals,
                            mode="markers",
                            name="SAT",
                            marker=dict(symbol="triangle-down", color="#e74c3c", size=14,
                                        line=dict(color="white", width=1)),
                            hovertemplate="SAT<br>%{x|%d %b %Y}<br>%{y:,.0f} TL<extra></extra>",
                        ))

                    fig_rl.update_layout(
                        title=f"RL Ajanı Portfolio — {rl_symbol}",
                        template="plotly_dark",
                        xaxis_title="Tarih",
                        yaxis_title="Portföy Değeri (TL)",
                        height=450,
                        legend=dict(orientation="h", y=1.05),
                    )
                    st.plotly_chart(fig_rl, use_container_width=True)

                # ── İşlem listesi ───────────────────────────────────────────
                if trades:
                    st.subheader(f"📋 İşlemler ({len(trades)} adet)")
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
