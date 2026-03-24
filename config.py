# Tradebot v1 Yapılandırma Dosyası

# BIST50 Hisseleri (Yahoo Finance sembolleri ile .IS uzantılı)
BIST50_SYMBOLS = [
    "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS",
    "BIMAS.IS", "BRSAN.IS", "CINFO.IS", "CWENE.IS", "DOAS.IS",
    "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS",
    "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GESAN.IS", "GUBRF.IS",
    "GWIND.IS", "HALKB.IS", "HEKTS.IS", "ISCTR.IS", "ISGYO.IS",
    "ISMEN.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS", "KOZAL.IS",
    "KRDMD.IS", "MGROS.IS", "ODAS.IS",  "OYAKC.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS",  "SISE.IS",  "SMRTG.IS",
    "SOKM.IS",  "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS",
    "TOASO.IS", "TSKB.IS",  "TTKOM.IS", "TUPRS.IS", "VAKBN.IS",
    "YKBNK.IS", "ZOREN.IS"
] # Yaklasik 50 civari likit hisse

# Test için varsayılan periyot ve interval (Kısa vadeli trader test ayarları)
# "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
DEFAULT_PERIOD = "5d"    # Son 5 günün verisi
# "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
DEFAULT_INTERVAL = "5m"  # 5 dakikalık mumlar (Kısa vade/Test için)

# ============================================================
# SEKTÖR HARİTASI
# ============================================================
SECTOR_MAP = {
    # Bankalar / Finans / GYO
    "AKBNK.IS": "Banka", "GARAN.IS": "Banka", "ISCTR.IS": "Banka",
    "YKBNK.IS": "Banka", "HALKB.IS": "Banka", "VAKBN.IS": "Banka",
    "TSKB.IS": "Banka", "ISMEN.IS": "Banka", "EKGYO.IS": "GYO",
    "ISGYO.IS": "GYO", "SAHOL.IS": "Holding", "KCHOL.IS": "Holding",
    # Enerji / Petrokimya
    "TUPRS.IS": "Enerji", "PETKM.IS": "Enerji", "ODAS.IS": "Enerji",
    "CWENE.IS": "Enerji", "ENJSA.IS": "Enerji", "EUPWR.IS": "Enerji",
    "GESAN.IS": "Enerji", "GWIND.IS": "Enerji", "SMRTG.IS": "Enerji",
    "ZOREN.IS": "Enerji",
    # Ulasim / Havayolu
    "THYAO.IS": "Ulasim", "PGSUS.IS": "Ulasim", "TAVHL.IS": "Ulasim",
    # Otomotiv
    "FROTO.IS": "Otomotiv", "TOASO.IS": "Otomotiv", "DOAS.IS": "Otomotiv",
    "EGEEN.IS": "Otomotiv",
    # Demir-Celik / Agir Sanayi
    "EREGL.IS": "DemirCelik", "KRDMD.IS": "DemirCelik",
    "BRSAN.IS": "DemirCelik", "ARCLK.IS": "DayanikliTtuketim",
    # Madencilik
    "KOZAL.IS": "Madencilik", "KOZAA.IS": "Madencilik",
    # Insaat
    "ENKAI.IS": "Insaat", "OYAKC.IS": "Insaat", "TKFEN.IS": "Insaat",
    "KONTR.IS": "Insaat",
    # Savunma / Teknoloji
    "ASELS.IS": "Savunma", "ASTOR.IS": "Savunma",
    "CINFO.IS": "Teknoloji",
    # Kimya / Gubre
    "GUBRF.IS": "Kimya", "HEKTS.IS": "Kimya", "SASA.IS": "Kimya",
    # Perakende / Telekom
    "BIMAS.IS": "Perakende", "MGROS.IS": "Perakende", "SOKM.IS": "Perakende",
    "TCELL.IS": "Telekom", "TTKOM.IS": "Telekom",
    "SISE.IS": "Holding", "ALARK.IS": "Holding",
}

# ============================================================
# MAKRO VERİ SEMBÖLLERİ (yfinance)
# ============================================================
MACRO_SYMBOLS = {
    "petrol": "BZ=F",       # Brent Petrol
    "altin": "GC=F",        # Altin (USD/ons)
    "dolar": "USDTRY=X",    # USD/TRY kuru
    "bist100": "XU100.IS",  # BIST100 endeksi
}

# ============================================================
# SEKTÖR-MAKRO KORELASYON TABLOSU
# korelasyon: +1 = pozitif (makro yukselirse hisse de yukselir)
#            -1 = negatif (makro yukselirse hisse duser)
# ============================================================
SECTOR_MACRO_CORRELATION = {
    "Enerji":      {"petrol": +1, "dolar": +0.5},
    "Madencilik":  {"altin": +1, "dolar": +0.5},
    "Banka":       {"dolar": -1, "bist100": +1},
    "Ulasim":      {"dolar": +0.7, "petrol": -0.5, "bist100": +0.5},
    "Otomotiv":    {"dolar": +0.5, "bist100": +0.5},
    "DemirCelik":  {"dolar": +0.5, "bist100": +0.5},
    "Insaat":      {"dolar": -0.3, "bist100": +0.5},
    "Savunma":     {"dolar": +0.5, "bist100": +0.5},
    "Teknoloji":   {"bist100": +0.5},
    "Kimya":       {"dolar": +0.5, "petrol": +0.3},
    "Perakende":   {"dolar": -0.5, "bist100": +0.5},
    "Telekom":     {"bist100": +0.5},
    "Holding":     {"bist100": +1, "dolar": -0.3},
}

# ============================================================
# SEKTÖR - ENDEKS EŞLEŞTİRMESİ
# Sektör skorlaması için hisselerin ait olduğu sektörün kendi endeksi de analiz edilecek.
# ============================================================
SECTOR_INDEX_MAP = {
    "Banka": "XBANK.IS",
    "Holding": "XHOLD.IS",
    "Enerji": "XELKT.IS",     # Elektrik olarak yeralır
    "Ulasim": "XULAS.IS",
    "Otomotiv": "XMESY.IS",   # Metal Eşya, Makine (Oto dahil)
    "DemirCelik": "XUMAE.IS", # Metal Ana
    "Madencilik": "XMADN.IS",
    "Insaat": "XINSA.IS",
    "Savunma": "XMESY.IS",    # Metal Eşya/Makine
    "Teknoloji": "XUTEK.IS",
    "Kimya": "XKMYA.IS",
    "Perakende": "XTCRT.IS",
    "Telekom": "XILET.IS",
    "Gida": "XGIDA.IS"
}
