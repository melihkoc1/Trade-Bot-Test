"""
Tradebot V1 — Investing.com RSS Fetcher
========================================
Investing.com Türkiye'den RSS feed üzerinden haber çeker.
3 ayrı feed: Genel haberler, Borsa haberleri, Emtia haberleri.
"""

import feedparser
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from news.fetchers.base_fetcher import BaseFetcher


class InvestingFetcher(BaseFetcher):
    """Investing.com TR RSS haber çekici."""

    SOURCE_NAME = "investing"

    RSS_FEEDS = {
        "investing_genel": "https://tr.investing.com/rss/news.rss",
        "investing_borsa": "https://tr.investing.com/rss/news_25.rss",
        "investing_emtia": "https://tr.investing.com/rss/news_11.rss",
        "investing_forex": "https://tr.investing.com/rss/news_1.rss",
    }

    # Genişletilmiş sembol eşanlamlıları
    SYMBOL_SYNONYMS = {
        "THYAO.IS": [
            "THY",
            "TURK HAVA YOLLARI",
            "TÜRK HAVA YOLLARI",
            "TURKISH AIRLINES",
        ],
        "ASELS.IS": ["ASELSAN", "ASELSAN ELEKTRONIK"],
        "EREGL.IS": ["EREĞLİ", "EREGLI", "ERDEMİR", "EREGLI DEMIR CELIK"],
        "KCHOL.IS": ["KOÇ HOLDİNG", "KOC HOLDING", "KOÇ"],
        "SAHOL.IS": ["SABANCI HOLDİNG", "SABANCI HOLDING", "SABANCI"],
        "SISE.IS": ["ŞİŞECAM", "SISECAM", "ŞİŞE CAM"],
        "AKBNK.IS": ["AKBANK"],
        "GARAN.IS": ["GARANTİ", "GARANTI BBVA", "GARANTİ BANKASI"],
        "TUPRS.IS": ["TÜPRAŞ", "TUPRAS"],
        "FROTO.IS": ["FORD OTOSAN", "FORD"],
        "TOASO.IS": ["TOFAŞ", "TOFAS"],
        "PGSUS.IS": ["PEGASUS"],
        "TAVHL.IS": ["TAV", "TAV HAVALIMANLARI"],
        "BIMAS.IS": ["BİM", "BIM", "BİM MAĞAZALARI"],
        "MGROS.IS": ["MİGROS", "MIGROS"],
        "SOKM.IS": ["ŞOK", "SOK", "ŞOK MARKET"],
        "TCELL.IS": ["TURKCELL"],
        "TTKOM.IS": ["TÜRK TELEKOM", "TURK TELEKOM"],
        "HALKB.IS": ["HALKBANK", "HALK BANKASI"],
        "VAKBN.IS": ["VAKIFBANK", "VAKIF BANKASI"],
        "ISCTR.IS": ["İŞ BANKASI", "IS BANKASI", "ISBANK"],
        "YKBNK.IS": ["YAPI KREDİ", "YAPI KREDI"],
        "KOZAL.IS": ["KOZA ALTIN"],
        "KOZAA.IS": ["KOZA ANADOLU"],
        "KRDMD.IS": ["KARDEMİR", "KARDEMIR"],
        "PETKM.IS": ["PETKİM", "PETKIM"],
        "SASA.IS": ["SASA POLYESTER", "SASA"],
        "ENKAI.IS": ["ENKA", "ENKA İNŞAAT"],
        "ARCLK.IS": ["ARÇELİK", "ARCELIK"],
        "DOAS.IS": ["DOĞUŞ OTOMOTİV", "DOGUS OTOMOTIV"],
        "GUBRF.IS": ["GÜBRE FABRİKALARI", "GUBRE FABRIKALARI"],
        "HEKTS.IS": ["HEKTAŞ", "HEKTAS"],
        "ASTOR.IS": ["ASELSAN ASTOR"],
        "EKGYO.IS": ["EMLAK KONUT"],
    }

    # KAP anahtar kelimeleri
    KAP_KEYWORDS = [
        "KAP:",
        "KAP ",
        "KAMUYU AYDINLATMA",
        "ÖZEL DURUM AÇIKLAMASI",
        "BİLDİRİM",
        "SPK",
        "SERMAYE PİYASASI",
        "TEMETTÜ",
        "TEMETTU",
        "BEDELSİZ",
        "BEDELLİ",
        "SERMAYE ARTIRIMI",
        "KAR DAĞITIM",
    ]

    # Makro haber anahtar kelimeleri
    MACRO_KEYWORDS = {
        "faiz": [
            "FAİZ",
            "FAIZ",
            "MERKEZ BANKASI",
            "TCMB",
            "FED",
            "ECB",
            "POLITIKA FAIZI",
            "PARA POLİTİKASI",
        ],
        "enflasyon": ["ENFLASYON", "TÜFE", "TUFE", "FİYAT ARTIŞI", "FIYAT ARTISI"],
        "doviz": ["DOLAR", "EURO", "DÖVİZ", "DOVIZ", "USD", "EUR", "USD/TRY", "KUR"],
        "petrol": ["PETROL", "BRENT", "OPEC", "CRUDE", "HAM PETROL"],
        "altin": ["ALTIN", "GOLD", "OUNCE", "ONS"],
        "büyüme": ["BÜYÜME", "GSYİH", "GSYIH", "GDP", "EKONOMİK BÜYÜME"],
    }

    def fetch(
        self, symbol: str = None, sector: str = None, category: str = None
    ) -> list[dict]:
        """Sembol veya sektör bazlı haber çek."""
        all_news = self.fetch_all()

        if not symbol and not sector:
            return all_news

        # Arama terimlerini belirle
        search_terms = []
        if symbol:
            clean = symbol.replace(".IS", "").upper()
            search_terms = [clean]
            full_symbol = symbol if ".IS" in symbol else f"{symbol}.IS"
            if full_symbol in self.SYMBOL_SYNONYMS:
                search_terms.extend(self.SYMBOL_SYNONYMS[full_symbol])
        elif sector:
            # Sektör anahtar kelimeleri classifier'dan gelecek
            search_terms = [sector.upper()]

        if not search_terms:
            return all_news

        # Filtrele
        filtered = []
        for item in all_news:
            text = f"{item['title']} {item['summary']}".upper()
            for term in search_terms:
                if term.upper() in text:
                    # Sembol eşleşmesi bulundu
                    if symbol:
                        full_sym = symbol if ".IS" in symbol else f"{symbol}.IS"
                        item["symbols"] = [full_sym]
                        item["news_type"] = "kap" if item.get("_is_kap") else "hisse"
                    elif sector:
                        item["news_type"] = "sektor"
                        item["category"] = sector
                    filtered.append(item)
                    break

        return filtered

    def fetch_all(self) -> list[dict]:
        """Tüm RSS feed'lerden haberleri çek."""
        all_news = []
        seen_urls = set()

        for feed_name, feed_url in self.RSS_FEEDS.items():
            response = self._safe_get(feed_url)
            if not response:
                continue

            try:
                feed = feedparser.parse(response.content)
                for entry in feed.entries:
                    url = entry.get("link", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    full_text = f"{title} {summary}".upper()

                    # Yayın tarihini parse et
                    published_at = self._parse_date(entry.get("published"))

                    # KAP tespiti
                    is_kap = any(kw.upper() in full_text for kw in self.KAP_KEYWORDS)

                    # Makro haber tespiti
                    macro_type = self._detect_macro_type(full_text)

                    # Haber tipi belirle
                    if is_kap:
                        news_type = "kap"
                    elif macro_type:
                        news_type = "makro"
                    elif "borsa" in feed_name:
                        news_type = "hisse"
                    elif "emtia" in feed_name or "forex" in feed_name:
                        news_type = "makro"
                    else:
                        news_type = "genel"

                    # Sembol tespiti (otomatik)
                    detected_symbols = self._detect_symbols(full_text)

                    item = self._make_news_item(
                        title=title,
                        url=url,
                        summary=summary,
                        published_at=published_at,
                        news_type=news_type,
                        category=macro_type,
                        symbols=detected_symbols,
                        raw_data={"feed": feed_name, "is_kap": is_kap},
                    )
                    item["_is_kap"] = is_kap  # İç kullanım
                    all_news.append(item)

            except Exception as e:
                print(f"[{self.SOURCE_NAME}] RSS parse hatası ({feed_name}): {e}")

        # Tarihe göre sırala
        all_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return all_news

    def _detect_symbols(self, text: str) -> list[str]:
        """Metin içinden otomatik sembol tespiti."""
        detected = []
        text_upper = text.upper()
        for symbol, synonyms in self.SYMBOL_SYNONYMS.items():
            clean = symbol.replace(".IS", "")
            all_terms = [clean] + [s.upper() for s in synonyms]
            for term in all_terms:
                if term in text_upper:
                    detected.append(symbol)
                    break
        return detected

    def _detect_macro_type(self, text: str) -> str | None:
        """Makro haber tipini tespit et."""
        text_upper = text.upper()
        for macro_type, keywords in self.MACRO_KEYWORDS.items():
            if any(kw in text_upper for kw in keywords):
                return macro_type
        return None

    def _parse_date(self, date_str: str) -> str | None:
        """RSS tarih formatını ISO formatına çevir."""
        if not date_str:
            return datetime.now().isoformat()
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.isoformat()
        except Exception:
            try:
                # Alternatif format denemesi
                for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        return datetime.strptime(date_str, fmt).isoformat()
                    except ValueError:
                        continue
            except Exception:
                pass
            return datetime.now().isoformat()
