"""
Tradebot V1 — Haber Yönetim Merkezi (News Manager)
====================================================
Tüm haber kaynaklarını, duygu analizini, veritabanını ve
makro etki hesaplamalarını orkestre eden ana sınıf.

strategy.py ve app.py ile geriye dönük uyumlu arayüz sağlar.
"""

import time
from typing import Optional
from config import SECTOR_MAP

from news.database import NewsDatabase
from news.classifier import NewsClassifier
from news.macro_impact import MacroImpactCalculator
from news.sentiment import SentimentEnsemble
from news.fetchers.investing_fetcher import InvestingFetcher
from news.fetchers.kap_fetcher import KAPFetcher
from news.fetchers.bigpara_fetcher import BigparaFetcher
from news.fetchers.foreks_fetcher import ForeksFetcher
from news.fetchers.tcmb_fetcher import TCMBFetcher


class NewsManager:
    """
    Haber analizi orkestrasyon sınıfı.

    Kullanım:
        from news import NewsManager
        nm = NewsManager()
        score, news_items = nm.get_sentiment_score("THYAO.IS")
    """

    def __init__(self, use_db: bool = True):
        # Veritabanı
        self.db = NewsDatabase() if use_db else None

        # Fetcher'lar
        self.fetchers = {
            "investing": InvestingFetcher(),
            "kap": KAPFetcher(),
            "bigpara": BigparaFetcher(),
            "foreks": ForeksFetcher(),
            "tcmb": TCMBFetcher(),
        }

        # Analiz modülleri
        self.sentiment = SentimentEnsemble()
        self.classifier = NewsClassifier()
        self.macro_calc = MacroImpactCalculator()

        # Cache (bellekte)
        self._cache = {}
        self._cache_ttl = 600  # 10 dakika

    # ──────────────────────────────────────────────────
    # ANA API (strategy.py uyumlu)
    # ──────────────────────────────────────────────────

    def get_sentiment_score(self, symbol: str) -> tuple:
        """
        strategy.py ve app.py ile uyumlu ana fonksiyon.
        Eski news_scraper.get_sentiment_score() yerine geçer.

        Returns:
            (normalized_score, news_items_list)
            normalized_score: -50 ile +50 arası float
            news_items_list: [{title, source, sentiment, score, type, link, ...}, ...]
        """
        # Cache kontrolü
        cache_key = f"sentiment_{symbol}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        # 1. Hisseye özel haberleri çek (tüm kaynaklardan)
        news_items = self._fetch_for_symbol(symbol)

        # 2. Yoksa sektör haberleri
        is_sector_news = False
        if not news_items:
            sector = SECTOR_MAP.get(symbol)
            if sector:
                news_items = self._fetch_for_sector(sector)
                is_sector_news = True

        # 3. Yoksa genel piyasa haberleri
        if not news_items:
            news_items = self._fetch_general()
            if not news_items:
                return 0, []

        # 4. Duygu analizi (en güncel 5 haberi analiz et)
        analyzed_news = []
        total_score = 0
        count = 0

        for item in news_items[:5]:
            sentiment = self.sentiment.analyze(
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                symbol=symbol,
                news_type=item.get("news_type", "genel"),
            )

            # Eski format uyumu (app.py gösterimi için)
            compat_item = {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "link": item.get("url", ""),
                "source": item.get("source", ""),
                "published": item.get("published_at", ""),
                "type": self._map_news_type(item.get("news_type", "genel")),
                "is_kap": item.get("news_type") == "kap",
                "score": sentiment["score"],
                "sentiment": self._map_sentiment_label(sentiment["label"]),
                "confidence": sentiment.get("confidence", 0),
                "model": sentiment.get("model", ""),
            }
            analyzed_news.append(compat_item)

            total_score += sentiment["score"]
            count += 1

            # Veritabanına kaydet
            if self.db:
                try:
                    news_id = self.db.insert_news(
                        source=item.get("source", ""),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        summary=item.get("summary"),
                        published_at=item.get("published_at"),
                        news_type=item.get("news_type", "genel"),
                        category=item.get("category"),
                    )
                    if news_id:
                        self.db.insert_sentiment(
                            news_id=news_id,
                            model=sentiment.get("model", "ensemble"),
                            score=sentiment["score"],
                            confidence=sentiment.get("confidence", 0),
                            label=sentiment["label"],
                        )
                        self.db.link_news_to_symbol(news_id, symbol)
                except Exception:
                    pass  # DB hatası sessizce geçilir

        # 5. Normalleştirme
        avg_score = total_score / count if count > 0 else 50
        normalized_score = avg_score - 50

        # Sektör haberi ise etkiyi azalt (%50)
        if is_sector_news:
            normalized_score *= 0.5

        # 6. Makro etki bonus
        macro_bonus = self._calculate_macro_bonus(symbol)
        normalized_score += macro_bonus

        # Sınırla
        normalized_score = max(-50, min(50, normalized_score))

        result = (round(normalized_score, 1), analyzed_news)
        self._set_cache(cache_key, result)
        return result

    # ──────────────────────────────────────────────────
    # Haber Çekme
    # ──────────────────────────────────────────────────

    def _fetch_for_symbol(self, symbol: str) -> list[dict]:
        """Bir sembol için tüm kaynaklardan haber çek."""
        all_news = []

        # Önce veritabanından kontrol (cache)
        if self.db:
            db_news = self.db.get_news_by_symbol(symbol, hours=6)
            if db_news:
                return db_news[:10]

        # Tüm fetcher'lardan çek
        for name, fetcher in self.fetchers.items():
            try:
                news = fetcher.fetch(symbol=symbol)
                if news:
                    # Sınıflandır ve zenginleştir
                    news = self.classifier.classify_batch(news)
                    all_news.extend(news)
            except Exception as e:
                print(f"[NewsManager] {name} fetch hatası: {e}")

        # Deduplicate by URL
        seen = set()
        unique = []
        for item in all_news:
            url = item.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(item)

        # Tarihe göre sırala
        unique.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return unique[:10]

    def _fetch_for_sector(self, sector: str) -> list[dict]:
        """Sektör bazlı haber çek."""
        all_news = []

        for name, fetcher in self.fetchers.items():
            try:
                news = fetcher.fetch(sector=sector)
                if news:
                    for item in news:
                        item["news_type"] = "sektor"
                        item["category"] = sector
                    all_news.extend(news)
            except Exception as e:
                print(f"[NewsManager] {name} sektör fetch hatası: {e}")

        seen = set()
        unique = []
        for item in all_news:
            url = item.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(item)

        unique.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return unique[:10]

    def _fetch_general(self) -> list[dict]:
        """Genel piyasa haberleri çek."""
        # Sadece Investing ve Foreks'ten genel haberler
        all_news = []
        for name in ["investing", "foreks"]:
            try:
                news = self.fetchers[name].fetch_all()
                all_news.extend(news[:10])
            except Exception:
                pass

        all_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return all_news[:10]

    def fetch_all_sources(self) -> dict:
        """Tüm kaynaklardan haberleri çek ve veritabanına kaydet."""
        results = {}
        for name, fetcher in self.fetchers.items():
            try:
                news = fetcher.fetch_all()
                # Sınıflandır
                news = self.classifier.classify_batch(news)
                # Veritabanına kaydet
                if self.db:
                    inserted = self.db.insert_news_batch(news)
                    results[name] = {"fetched": len(news), "inserted": inserted}
                else:
                    results[name] = {"fetched": len(news), "inserted": 0}
            except Exception as e:
                results[name] = {"fetched": 0, "error": str(e)}
                print(f"[NewsManager] {name} toplu fetch hatası: {e}")

        return results

    # ──────────────────────────────────────────────────
    # Makro Etki
    # ──────────────────────────────────────────────────

    def _calculate_macro_bonus(self, symbol: str) -> float:
        """Sembol için makro haber etkisi hesapla."""
        if not self.db:
            return 0.0

        sector = SECTOR_MAP.get(symbol)
        if not sector:
            return 0.0

        try:
            macro_news = self.db.get_news_by_type("makro", hours=12)
            if not macro_news:
                return 0.0

            total_impact = 0.0
            count = 0

            for news in macro_news[:5]:
                sentiments = self.db.get_sentiment_for_symbol(symbol, hours=12)
                if not sentiments:
                    continue

                avg_sentiment = sum(s["score"] for s in sentiments) / len(sentiments)
                impacts = self.macro_calc.calculate_impact(news, avg_sentiment - 50)
                symbol_impact = self.macro_calc.get_impact_for_symbol(symbol, impacts)

                if symbol_impact != 0:
                    total_impact += symbol_impact
                    count += 1

            if count > 0:
                return round(
                    total_impact / count * 10, 1
                )  # -10 ile +10 arası normalize
        except Exception:
            pass

        return 0.0

    def get_macro_summary(self) -> dict:
        """Genel makro haber özeti ve sektör etkileri."""
        if not self.db:
            return {}

        macro_news = self.db.get_news_by_type("makro", hours=24)
        if not macro_news:
            return {"news_count": 0, "sector_impacts": {}}

        sector_impacts = {}
        for news in macro_news[:10]:
            # Basit sentiment hesaplama
            sentiment = self.sentiment.analyze(
                title=news.get("title", ""),
                summary=news.get("summary", ""),
                news_type="makro",
            )
            impacts = self.macro_calc.calculate_impact(news, sentiment["score"] - 50)
            for impact in impacts:
                sector = impact["sector"]
                if sector not in sector_impacts:
                    sector_impacts[sector] = []
                sector_impacts[sector].append(impact["impact_score"])

        # Ortalama al
        avg_impacts = {
            sector: round(sum(scores) / len(scores), 3)
            for sector, scores in sector_impacts.items()
        }

        return {
            "news_count": len(macro_news),
            "sector_impacts": avg_impacts,
        }

    # ──────────────────────────────────────────────────
    # Yardımcılar
    # ──────────────────────────────────────────────────

    @staticmethod
    def _map_news_type(news_type: str) -> str:
        """Dahili tip → UI etiket dönüşümü."""
        mapping = {
            "kap": "KAP",
            "hisse": "Haber",
            "sektor": "Sektör",
            "makro": "Makro",
            "genel": "Haber",
        }
        return mapping.get(news_type, "Haber")

    @staticmethod
    def _map_sentiment_label(label: str) -> str:
        """Dahili etiket → UI uyumlu İngilizce etiket (3-class)."""
        mapping = {
            "pozitif": "Positive",
            "notr": "Neutral",
            "negatif": "Negative",
        }
        return mapping.get(label, "Neutral")

    def _get_cache(self, key: str):
        """Cache'den oku."""
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value):
        """Cache'e yaz."""
        self._cache[key] = (value, time.time())

    def clear_cache(self):
        """Cache'i temizle."""
        self._cache.clear()

    def get_stats(self) -> dict:
        """Sistem istatistikleri."""
        stats = {
            "engines": self.sentiment.engines_status,
            "fetchers": list(self.fetchers.keys()),
            "cache_size": len(self._cache),
        }
        if self.db:
            stats["database"] = self.db.get_stats()
        return stats


# ──────────────────────────────────────────────────
# Singleton + geriye dönük uyumlu fonksiyon
# ──────────────────────────────────────────────────

_manager: Optional[NewsManager] = None


def get_manager() -> NewsManager:
    """Singleton NewsManager instance."""
    global _manager
    if _manager is None:
        _manager = NewsManager()
    return _manager


def get_sentiment_score(symbol: str) -> tuple:
    """
    news_scraper.get_sentiment_score() ile tam uyumlu.
    strategy.py ve app.py bu fonksiyonu kullanır.
    """
    return get_manager().get_sentiment_score(symbol)
