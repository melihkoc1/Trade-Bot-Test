"""
Tradebot V1 — Haber Sınıflandırıcı (Classifier)
=================================================
Haberleri hisse, sektör, makro ve genel olarak sınıflandırır.
Haber → sembol eşleştirmesini yapar.
"""

import re
from config import SECTOR_MAP, BIST50_SYMBOLS


class NewsClassifier:
    """Haber → hisse/sektör/makro eşleştirme motoru."""

    # Sektör anahtar kelimeleri (genişletilmiş)
    SECTOR_KEYWORDS = {
        "Banka": [
            "BANKA",
            "AKBANK",
            "GARANTİ",
            "GARANTI",
            "İŞ BANKASI",
            "IS BANKASI",
            "YAPI KREDİ",
            "YAPI KREDI",
            "VAKIFBANK",
            "HALKBANK",
            "KREDİ",
            "KREDI",
            "FAİZ",
            "FAIZ",
            "MEVDUAT",
            "BANKACILIK",
        ],
        "Enerji": [
            "ENERJİ",
            "ENERJI",
            "ELEKTRİK",
            "ELEKTRIK",
            "PETROL",
            "DOĞALGAZ",
            "DOGALGAZ",
            "GÜNEŞ",
            "GUNES",
            "RÜZGAR",
            "RUZGAR",
            "TÜPRAŞ",
            "TUPRAS",
            "YENİLENEBİLİR",
            "YENILENEBILIR",
        ],
        "Ulasim": [
            "ULAŞIM",
            "ULASIM",
            "HAVA YOLU",
            "THY",
            "PEGASUS",
            "TAV",
            "LOJİSTİK",
            "LOJISTIK",
            "HAVAALANI",
            "HAVALIMANL",
        ],
        "Otomotiv": [
            "OTOMOTİV",
            "OTOMOTIV",
            "ARAÇ",
            "ARAC",
            "ÜRETİM",
            "FORD",
            "TOFAŞ",
            "TOFAS",
            "DOĞUŞ",
            "DOGUS",
            "LASTİK",
            "LASTIK",
        ],
        "DemirCelik": [
            "DEMİR",
            "DEMIR",
            "ÇELİK",
            "CELIK",
            "EREĞLİ",
            "EREGLI",
            "KARDEMİR",
            "KARDEMIR",
            "METAL",
            "CEVHERİ",
            "CEVHERI",
        ],
        "Holding": [
            "HOLDİNG",
            "HOLDING",
            "KOÇ",
            "KOC",
            "SABANCI",
            "ŞİŞECAM",
            "SISECAM",
            "ALARKO",
            "İŞTİRAK",
            "ISTIRAK",
        ],
        "Madencilik": [
            "MADENCİLİK",
            "MADENCLK",
            "KOZA",
            "ALTIN",
            "MADEN",
        ],
        "Insaat": [
            "İNŞAAT",
            "INSAAT",
            "ENKA",
            "TKFEN",
            "KONTR",
            "OYAK",
            "MÜTEAHHİT",
            "MUTEAHHIT",
            "KONUT",
            "YAPI",
        ],
        "Savunma": [
            "SAVUNMA",
            "ASELSAN",
            "ASTOR",
            "SİLAH",
            "SILAH",
            "MİLİTER",
            "MILITER",
            "HAVA SAVUNMA",
        ],
        "Teknoloji": [
            "TEKNOLOJİ",
            "TEKNOLOJI",
            "YAZILIM",
            "BİLİŞİM",
            "BILISIM",
            "ÇİP",
            "CIP",
            "DİJİTAL",
            "DIJITAL",
        ],
        "Kimya": [
            "KİMYA",
            "KIMYA",
            "GÜBRE",
            "GUBRE",
            "POLİMER",
            "POLIMER",
            "PETROKİMYA",
            "PETROKIMYA",
            "PLASTİK",
            "PLASTIK",
        ],
        "Perakende": [
            "PERAKENDE",
            "MARKET",
            "BİM",
            "BIM",
            "MİGROS",
            "MIGROS",
            "ŞOK",
            "SOK",
            "GIDA",
            "TÜKETİM",
            "TUKETIM",
        ],
        "Telekom": [
            "TELEKOM",
            "TURKCELL",
            "İLETİŞİM",
            "ILETISIM",
            "GSM",
            "FİBER",
            "FIBER",
            "5G",
        ],
        "GYO": [
            "GAYRİMENKUL",
            "GAYRIMENKUL",
            "GYO",
            "EMLAK",
            "KONUT",
            "REİT",
            "REIT",
        ],
        "DayanikliTtuketim": [
            "DAYANIKLI TÜKETİM",
            "ARÇELİK",
            "ARCELIK",
            "BEYAZ EŞYA",
            "BEYAZ ESYA",
            "ELEKTRONİK",
        ],
    }

    # Makro anahtar kelimeler (detaylı)
    MACRO_KEYWORDS = {
        "faiz": [
            "FAİZ",
            "FAIZ",
            "MERKEZ BANKASI",
            "TCMB",
            "FED",
            "ECB",
            "POLİTİKA FAİZİ",
            "PARA POLİTİKASI",
            "FAİZ İNDİRİMİ",
            "FAİZ ARTIŞI",
        ],
        "enflasyon": [
            "ENFLASYON",
            "TÜFE",
            "TUFE",
            "ÜFE",
            "UFE",
            "FİYAT ARTIŞI",
            "FIYAT ARTISI",
            "HAYAT PAHALILIĞI",
        ],
        "doviz": [
            "DOLAR",
            "EURO",
            "DÖVİZ",
            "DOVIZ",
            "USD/TRY",
            "EUR/TRY",
            "KUR",
            "DÖVİZ KURU",
            "DOVIZ KURU",
        ],
        "petrol": [
            "PETROL",
            "BRENT",
            "OPEC",
            "CRUDE",
            "HAM PETROL",
            "OPEC+",
        ],
        "altin": [
            "ALTIN",
            "GOLD",
            "ONS",
            "OUNCE",
            "KIYMETLİ MADEN",
        ],
        "buyume": [
            "BÜYÜME",
            "BUYUME",
            "GSYİH",
            "GSYIH",
            "GDP",
            "EKONOMİK BÜYÜME",
            "RESESYON",
        ],
        "istihdam": [
            "İSTİHDAM",
            "ISTIHDAM",
            "İŞSİZLİK",
            "ISSIZLIK",
            "TARIM DIŞI İSTİHDAM",
        ],
    }

    def __init__(self):
        # Ters sembol haritası: sembol adı → tam sembol
        self._symbol_lookup = {}
        for sym in BIST50_SYMBOLS:
            clean = sym.replace(".IS", "")
            self._symbol_lookup[clean] = sym

    def classify(self, news_item: dict) -> dict:
        """
        Bir haberi sınıflandır ve zenginleştir.

        Returns:
            Güncellenmiş news_item (news_type, category, symbols eklenir)
        """
        title = news_item.get("title", "")
        summary = news_item.get("summary", "")
        full_text = f"{title} {summary}".upper()

        # 1. Hisse tespiti (en spesifik → en yüksek öncelik)
        detected_symbols = self._detect_symbols(full_text)
        if detected_symbols:
            news_item["symbols"] = list(
                set(news_item.get("symbols", []) + detected_symbols)
            )
            if news_item.get("news_type") not in ("kap",):
                news_item["news_type"] = "hisse"

        # 2. Makro haber tespiti
        macro_type = self._detect_macro(full_text)
        if macro_type and not detected_symbols:
            news_item["news_type"] = "makro"
            news_item["category"] = macro_type

        # 3. Sektör tespiti
        detected_sector = self._detect_sector(full_text)
        if detected_sector and not detected_symbols and not macro_type:
            news_item["news_type"] = "sektor"
            news_item["category"] = detected_sector

        return news_item

    def classify_batch(self, news_list: list[dict]) -> list[dict]:
        """Toplu sınıflandırma."""
        return [self.classify(item) for item in news_list]

    def find_related_symbols(self, news_item: dict) -> list[dict]:
        """
        Bir haber için ilişkili sembolleri bul.

        Returns:
            [{"symbol": "THYAO.IS", "relevance_score": 0.9, "match_type": "direct"}, ...]
        """
        title = news_item.get("title", "")
        summary = news_item.get("summary", "")
        full_text = f"{title} {summary}".upper()
        results = []

        # Direkt sembol eşleşmesi
        for clean_sym, full_sym in self._symbol_lookup.items():
            if clean_sym in full_text:
                results.append(
                    {
                        "symbol": full_sym,
                        "relevance_score": 1.0,
                        "match_type": "direct",
                    }
                )

        # Sektör bazlı eşleşme (daha düşük relevance)
        sector = self._detect_sector(full_text)
        if sector:
            for sym, sec in SECTOR_MAP.items():
                if sec == sector and sym not in [r["symbol"] for r in results]:
                    results.append(
                        {
                            "symbol": sym,
                            "relevance_score": 0.4,
                            "match_type": "sector",
                        }
                    )

        return results

    def _detect_symbols(self, text: str) -> list[str]:
        """Metin içinden BIST50 sembollerini tespit et."""
        detected = []
        for clean_sym, full_sym in self._symbol_lookup.items():
            # Tam kelime eşleşmesi (false positive önleme)
            pattern = rf"\b{re.escape(clean_sym)}\b"
            if re.search(pattern, text):
                detected.append(full_sym)
        return detected

    def _detect_sector(self, text: str) -> str | None:
        """Sektör tespiti."""
        best_sector = None
        best_count = 0

        for sector, keywords in self.SECTOR_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_sector = sector

        return best_sector if best_count >= 1 else None

    def _detect_macro(self, text: str) -> str | None:
        """Makro haber tipi tespiti."""
        for macro_type, keywords in self.MACRO_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return macro_type
        return None
