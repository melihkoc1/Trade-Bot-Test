"""
Piotroski F-Score + Temel Analiz — BIST hisseleri için.

Tüm oranlar ham finansal tablolardan (TL→TL) hesaplanır.
info dict'indeki hazır oranlar kullanılmaz (BIST için güvenilmez).

F-Score 9 kriterden oluşur:
  Karlılık   (4): ROA pozitif, CFO pozitif, ROA arttı, Tahakkuk kalitesi
  Kaldıraç   (3): Borç azaldı, Current ratio arttı, Yeni hisse ihraç yok
  Verimlilik (2): Brüt marj arttı, Varlık devir hızı arttı

Toplam: 0-2 Zayıf | 3-5 Nötr | 6-7 İyi | 8-9 Güçlü
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ── Non-Streamlit önbellek (rl_backtester vb. için) ───────────────────────────
_fscore_cache_raw: dict = {}  # {symbol: score_int_or_None}


def get_fscore_filter(symbol: str) -> int | None:
    """
    Piotroski F-Score'u döndürür (0-9).
    Streamlit context dışından (rl_backtester, CLI) çağrılabilir.
    Sonuçlar modül ömrü boyunca bellekte tutulur.
    """
    if symbol in _fscore_cache_raw:
        return _fscore_cache_raw[symbol]
    res = get_piotroski(symbol)
    score = res.get("score")
    _fscore_cache_raw[symbol] = score
    return score


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _get(df: pd.DataFrame, *keys) -> float | None:
    """DataFrame'den ilk bulunan satırı döndür (yfinance sürüm farklılıklarına karşı)."""
    if df is None or df.empty:
        return None
    for key in keys:
        matches = [r for r in df.index if key.lower() in str(r).lower()]
        if matches:
            row = df.loc[matches[0]]
            vals = row.dropna().values
            return float(vals[0]) if len(vals) > 0 else None
    return None


def _get_prev(df: pd.DataFrame, *keys) -> float | None:
    """İkinci sütunu (önceki yıl) döndür."""
    if df is None or df.empty or df.shape[1] < 2:
        return None
    for key in keys:
        matches = [r for r in df.index if key.lower() in str(r).lower()]
        if matches:
            row = df.loc[matches[0]]
            vals = row.dropna().values
            return float(vals[1]) if len(vals) > 1 else None
    return None


def _safe(v, digits=2):
    """None → None, aksi hâlde yuvarlat."""
    return round(v, digits) if v is not None else None


def _growth(curr, prev):
    """YoY büyüme oranı (%). None-safe."""
    if curr is not None and prev and prev != 0:
        return round((curr - prev) / abs(prev) * 100, 1)
    return None


def _mn(v):
    """TL cinsinden mn formatla (örn. 1.234.567 → 1,234.6 mn TL)."""
    if v is None:
        return "—"
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f} Tr TL"
    if abs(v) >= 1e9:
        return f"{v/1e9:.1f} Mr TL"
    if abs(v) >= 1e6:
        return f"{v/1e6:.0f} mn TL"
    return f"{v:.0f} TL"


def _pct(v):
    return f"%{v*100:.1f}" if v is not None else "—"


def _ratio(v, digits=2):
    return f"{v:.{digits}f}x" if v is not None else "—"


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _get_usdtry() -> float | None:
    """USDTRY kurunu önbellekle (1 saat). Tüm hisse sorgularında paylaşılır."""
    try:
        hist = yf.Ticker("USDTRY=X").history(period="5d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_piotroski(symbol: str) -> dict:
    """
    Piotroski F-Score + tam temel analiz metriklerini döndürür.
    Tüm oranlar ham finansal tablolardan hesaplanır (TL-TL).
    """
    try:
        ticker = yf.Ticker(symbol)

        # ── Finansal tablolar ─────────────────────────────────────────
        inc  = ticker.financials
        bal  = ticker.balance_sheet
        cf   = ticker.cashflow
        info = ticker.info or {}

        if inc is None or inc.empty:
            inc = ticker.income_stmt
        if cf is None or cf.empty:
            cf = ticker.cash_flow

        # ── Güncel fiyat (TRY) ve hisse sayısı ──────────────────────
        hist = ticker.history(period="5d")
        current_price_try = float(hist["Close"].iloc[-1]) if not hist.empty else None

        # Hisse sayısı — birden fazla kaynağa bak
        shares = (info.get("sharesOutstanding")
                  or info.get("impliedSharesOutstanding"))
        if not shares and current_price_try and info.get("marketCap"):
            # info["marketCap"] .IS hisseleri için TRY cinsinden gelir
            shares = info["marketCap"] / current_price_try

        # yfinance finansal tabloları USD cinsinden gelir (.IS hisseleri için).
        # Piyasa değerini de USD'a çevirerek tutarlı oran hesabı yapalım.
        usdtry = _get_usdtry()

        current_price_usd = (current_price_try / usdtry
                             if (current_price_try and usdtry) else None)

        # ── Ham veriler — Gelir Tablosu ───────────────────────────────
        revenue       = _get(inc, "Total Revenue", "Revenue")
        revenue_prev  = _get_prev(inc, "Total Revenue", "Revenue")
        gross_profit  = _get(inc, "Gross Profit")
        gross_profit_prev = _get_prev(inc, "Gross Profit")
        net_income    = _get(inc, "Net Income")
        net_income_prev = _get_prev(inc, "Net Income")
        op_income     = _get(inc, "Operating Income", "Ebit")

        # FAVÖK = Faaliyet Kârı + Amortisman
        da = _get(cf, "Depreciation", "Depreciation And Amortization",
                      "Depreciation Amortization Depletion")
        ebitda = ((op_income + da) if (op_income is not None and da is not None)
                  else _get(inc, "Normalized Ebitda", "Ebitda"))

        # ── Ham veriler — Bilanço ─────────────────────────────────────
        total_assets      = _get(bal, "Total Assets")
        total_assets_prev = _get_prev(bal, "Total Assets")

        equity      = _get(bal, "Total Stockholder Equity", "Stockholders Equity",
                               "Common Stock Equity", "Total Equity Gross Minority Interest")
        equity_prev = _get_prev(bal, "Total Stockholder Equity", "Stockholders Equity",
                                    "Common Stock Equity", "Total Equity Gross Minority Interest")

        # Toplam borç: uzun + kısa vadeli
        lt_debt  = _get(bal, "Long Term Debt",
                             "Long Term Debt And Capital Lease Obligation")
        lt_debt_prev = _get_prev(bal, "Long Term Debt",
                                      "Long Term Debt And Capital Lease Obligation")
        st_debt  = _get(bal, "Current Debt", "Short Long Term Debt",
                             "Current Debt And Capital Lease Obligation")
        total_debt = ((lt_debt or 0) + (st_debt or 0)
                      if (lt_debt is not None or st_debt is not None) else None)

        cash = _get(bal, "Cash And Cash Equivalents", "Cash",
                         "Cash Cash Equivalents And Short Term Investments")

        curr_assets      = _get(bal, "Current Assets", "Total Current Assets")
        curr_assets_prev = _get_prev(bal, "Current Assets", "Total Current Assets")
        curr_liab        = _get(bal, "Current Liabilities", "Total Current Liabilities",
                                     "Current Liab")
        curr_liab_prev   = _get_prev(bal, "Current Liabilities", "Total Current Liabilities",
                                         "Current Liab")

        # Hisse sayısı değişimi — bilanço common stock satırından
        cs      = _get(bal, "Common Stock")
        cs_prev = _get_prev(bal, "Common Stock")

        # ── Ham veriler — Nakit Akışı ─────────────────────────────────
        cfo      = _get(cf, "Total Cash From Operating", "Operating Cash Flow",
                             "Cash From Operations")
        cfo_prev = _get_prev(cf, "Total Cash From Operating", "Operating Cash Flow",
                                  "Cash From Operations")
        # capex yfinance'ta negatif gelir
        capex    = _get(cf, "Capital Expenditures", "Capital Expenditure",
                             "Purchase Of Property Plant And Equipment")

        # ── Türetilmiş değerler ───────────────────────────────────────
        avg_assets = (((total_assets or 0) + (total_assets_prev or total_assets or 0)) / 2
                      or 1)
        avg_assets_prev = total_assets_prev or total_assets or 1

        # Piyasa değeri USD — finansal tablolarla aynı para birimi
        market_cap_usd = (current_price_usd * shares
                          if (current_price_usd and shares) else None)
        # TRY cinsinden gösterim için de tutalım
        market_cap_try = (current_price_try * shares
                          if (current_price_try and shares) else None)
        market_cap = market_cap_usd  # oranlar USD/USD olacak
        net_debt   = ((total_debt - cash)
                      if (total_debt is not None and cash is not None) else None)
        ev         = ((market_cap + net_debt)
                      if (market_cap is not None and net_debt is not None) else None)
        fcf        = ((cfo + capex)           # capex < 0
                      if (cfo is not None and capex is not None) else None)

        # Temel oranlar — TL-TL
        pe = (market_cap / net_income
              if (market_cap and net_income and net_income > 0) else None)
        pb = (market_cap / equity
              if (market_cap and equity and equity > 0) else None)
        ev_ebitda = (ev / ebitda
                     if (ev and ebitda and ebitda > 0) else None)
        ev_sales  = (ev / revenue
                     if (ev and revenue and revenue > 0) else None)
        net_margin    = (net_income / revenue   if (net_income is not None and revenue) else None)
        ebitda_margin = (ebitda / revenue       if (ebitda is not None and revenue) else None)
        gross_margin  = (gross_profit / revenue if (gross_profit and revenue) else None)
        roe = (net_income / equity       if (net_income is not None and equity and equity > 0) else None)
        roa = (net_income / avg_assets   if net_income is not None else None)
        de  = (total_debt / equity       if (total_debt is not None and equity and equity > 0) else None)
        net_debt_ebitda = (net_debt / ebitda
                           if (net_debt is not None and ebitda and ebitda > 0) else None)
        curr_ratio      = (curr_assets / curr_liab
                           if (curr_assets and curr_liab) else None)

        # Piotroski için ek türetimler
        roa_prev        = (net_income_prev / avg_assets_prev
                           if net_income_prev is not None else None)
        cfo_ratio       = (cfo / avg_assets      if cfo is not None else None)
        lever           = (lt_debt / avg_assets  if lt_debt is not None else None)
        lever_prev      = (lt_debt_prev / avg_assets_prev if lt_debt_prev is not None else None)
        curr_ratio_prev = (curr_assets_prev / curr_liab_prev
                           if (curr_assets_prev and curr_liab_prev) else None)
        gross_margin_prev = (gross_profit_prev / revenue_prev
                             if (gross_profit_prev and revenue_prev) else None)
        asset_turn      = (revenue / avg_assets      if revenue else None)
        asset_turn_prev = (revenue_prev / avg_assets_prev if revenue_prev else None)

        # Büyüme oranları
        rev_growth = _growth(revenue, revenue_prev)
        ni_growth  = _growth(net_income, net_income_prev)

        # Temettü — sadece oran, para birimi yok → info dict burada güvenilir
        div_yield = (info.get("dividendYield") or 0) * 100

        # ── 9 Piotroski Kriteri ───────────────────────────────────────
        criteria = [
            # Karlılık
            {
                "group": "Karlılık",
                "name":  "ROA Pozitif",
                "passed": bool(roa is not None and roa > 0),
                "value":  _pct(roa),
                "desc":   "Net Kâr / Toplam Varlık > 0",
            },
            {
                "group": "Karlılık",
                "name":  "Faaliyet Nakit Akışı Pozitif",
                "passed": bool(cfo_ratio is not None and cfo_ratio > 0),
                "value":  _pct(cfo_ratio),
                "desc":   "CFO / Toplam Varlık > 0",
            },
            {
                "group": "Karlılık",
                "name":  "ROA Arttı",
                "passed": bool(roa is not None and roa_prev is not None and roa > roa_prev),
                "value":  f"{_pct(roa_prev)} → {_pct(roa)}",
                "desc":   "Bu yılın ROA'sı geçen yılı geçti",
            },
            {
                "group": "Karlılık",
                "name":  "Tahakkuk Kalitesi (CFO > Net Kâr)",
                "passed": bool(cfo is not None and net_income is not None and cfo > net_income),
                "value":  f"CFO {_mn(cfo)} / NK {_mn(net_income)}",
                "desc":   "Nakit kazanç, muhasebe kârından büyük",
            },
            # Kaldıraç & Likidite
            {
                "group": "Kaldıraç & Likidite",
                "name":  "Uzun Vadeli Borç Azaldı",
                "passed": bool(lever is not None and lever_prev is not None and lever < lever_prev),
                "value":  f"{_pct(lever_prev)} → {_pct(lever)}",
                "desc":   "UVB/Varlık oranı düştü",
            },
            {
                "group": "Kaldıraç & Likidite",
                "name":  "Cari Oran Arttı",
                "passed": bool(curr_ratio is not None and curr_ratio_prev is not None
                               and curr_ratio > curr_ratio_prev),
                "value":  (f"{curr_ratio_prev:.2f} → {curr_ratio:.2f}"
                           if (curr_ratio and curr_ratio_prev) else "—"),
                "desc":   "Kısa vadeli ödeme gücü iyileşti",
            },
            {
                "group": "Kaldıraç & Likidite",
                "name":  "Yeni Hisse İhraç Yok",
                "passed": bool(cs is not None and cs_prev is not None and cs <= cs_prev * 1.02),
                "value":  f"{_mn(cs_prev)} → {_mn(cs)}",
                "desc":   "Sermaye seyreltmesi yok (±%2 tolerans)",
            },
            # Operasyonel Verimlilik
            {
                "group": "Operasyonel Verimlilik",
                "name":  "Brüt Marj Arttı",
                "passed": bool(gross_margin is not None and gross_margin_prev is not None
                               and gross_margin > gross_margin_prev),
                "value":  f"{_pct(gross_margin_prev)} → {_pct(gross_margin)}",
                "desc":   "Brüt kâr marjı iyileşti",
            },
            {
                "group": "Operasyonel Verimlilik",
                "name":  "Varlık Devir Hızı Arttı",
                "passed": bool(asset_turn is not None and asset_turn_prev is not None
                               and asset_turn > asset_turn_prev),
                "value":  (f"{asset_turn_prev:.2f}x → {asset_turn:.2f}x"
                           if (asset_turn and asset_turn_prev) else "—"),
                "desc":   "Varlıkları daha verimli kullanıyor",
            },
        ]

        score = sum(1 for c in criteria if c["passed"])
        label = ("Güçlü" if score >= 8 else
                 "İyi"   if score >= 6 else
                 "Nötr"  if score >= 3 else "Zayıf")

        # ── Metrik grupları — UI için ─────────────────────────────────
        # TRY formatı için dönüşüm (gösterim amaçlı)
        def _mn_try(usd_val):
            """USD değeri TRY'ye çevirip formatla."""
            if usd_val is None:
                return "—"
            try_val = usd_val * usdtry if usdtry else usd_val
            return _mn(try_val)

        metrics = {
            "Değerleme": [
                ("Piyasa Değeri",  _mn_try(market_cap),     "Güncel fiyat × hisse sayısı (TL)"),
                ("F/K",            _safe(pe, 1),            "Fiyat / Hisse Başı Kâr"),
                ("PD/DD",          _safe(pb, 2),            "Piyasa Değeri / Özkaynaklar"),
                ("EV/FAVÖK",       _safe(ev_ebitda, 1),     "Firma Değeri / FAVÖK"),
                ("FD/Satışlar",    _safe(ev_sales, 2),      "Firma Değeri / Gelir"),
                ("Firma Değeri",   _mn_try(ev),             "PD + Net Borç (TL)"),
                ("Net Borç",       _mn_try(net_debt),       "Toplam Borç − Nakit (TL)"),
            ],
            "Kârlılık": [
                ("Gelir",          _mn_try(revenue),        "Son yıllık ciro (TL)"),
                ("FAVÖK",          _mn_try(ebitda),         "Faiz/Vergi/Amortisman öncesi kâr (TL)"),
                ("Net Kâr",        _mn_try(net_income),     "Son yıllık net kâr (TL)"),
                ("Brüt Marj",      _pct(gross_margin),      "Brüt Kâr / Gelir"),
                ("FAVÖK Marjı",    _pct(ebitda_margin),     "FAVÖK / Gelir"),
                ("Net Marj",       _pct(net_margin),        "Net Kâr / Gelir"),
                ("ROE",            _pct(roe),               "Net Kâr / Özkaynaklar"),
                ("ROA",            _pct(roa),               "Net Kâr / Toplam Varlık"),
            ],
            "Finansal Sağlık": [
                ("Borç/Özkaynak",  _safe(de, 2),            "Toplam Borç / Özkaynaklar"),
                ("Net Borç/FAVÖK", _safe(net_debt_ebitda,1),"Kaldıraç seviyesi"),
                ("Cari Oran",      _safe(curr_ratio, 2),    "Dönen Varlık / Kısa Vadeli Borç"),
                ("Serbest NK (FCF)",_mn_try(fcf),            "Faaliyet NK − Yatırım Harcaması (TL)"),
                ("Özkaynak",       _mn_try(equity),         "Defter değeri (TL)"),
            ],
            "Büyüme (YoY)": [
                ("Gelir Büyümesi", f"%{rev_growth}" if rev_growth is not None else "—",
                                   "Geçen yıla göre gelir artışı"),
                ("Net Kâr Büyümesi", f"%{ni_growth}" if ni_growth is not None else "—",
                                     "Geçen yıla göre net kâr artışı"),
                ("Temettü Verimi", f"%{div_yield:.2f}" if div_yield else "—",
                                   "Hisse başı temettü / fiyat"),
            ],
        }

        return {
            "score":    score,
            "criteria": criteria,
            "label":    label,
            "metrics":  metrics,
            "error":    None,
        }

    except Exception as e:
        return {
            "score":    None,
            "criteria": [],
            "label":    "Hata",
            "metrics":  {},
            "error":    str(e),
        }
