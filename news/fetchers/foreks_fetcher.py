"""
Tradebot V1 — ForeksHaber RSS Fetcher
======================================
Foreks haber ajansından anlık piyasa haberleri çeker.
"""

import feedparser
from datetime import datetime
from email.utils import parsedate_to_datetime
from news.fetchers.base_fetcher import BaseFetcher


class ForeksFetcher(BaseFetcher):
    """Foreks haber ajansı RSS çekici."""

    SOURCE_NAME = "foreks"
    REQUEST_DELAY = 0.5

    # Foreks haber RSS feed'leri
    RSS_FEEDS = {
        "foreks_genel": "https://www.foreks.com/feed/",
        "foreks_borsa": "https://www.foreks.com/category/borsa/feed/",
        "foreks_ekonomi": "https://www.foreks.com/category/ekonomi/feed/",
        "foreks_dunya": "https://www.foreks.com/category/dunya/feed/",
    }

    # Alternatif kaynaklar (Foreks erişilemezse)
    ALT_FEEDS = {
        "bloomberght": "https://www.bloomberght.com/rss",
        "paraanaliz": "https://www.paraanaliz.com/feed/",
    }

    def fetch(
        self, symbol: str = None, sector: str = None, category: str = None
    ) -> list[dict]:
        """Haber çek ve filtrele."""
        all_news = self.fetch_all()
        if not symbol:
            return all_news

        clean = symbol.replace(".IS", "").upper()
        filtered = []
        for item in all_news:
            text = f"{item['title']} {item['summary']}".upper()
            if clean in text:
                item["symbols"] = [symbol if ".IS" in symbol else f"{symbol}.IS"]
                item["news_type"] = "hisse"
                filtered.append(item)
        return filtered

    def fetch_all(self) -> list[dict]:
        """Tüm RSS feed'lerden haberleri çek."""
        all_news = []
        seen_urls = set()

        # Önce ana kaynak
        for feed_name, feed_url in self.RSS_FEEDS.items():
            news = self._parse_feed(feed_url, feed_name)
            for item in news:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_news.append(item)

        # Eğer ana kaynak boşsa, alternatif dene
        if len(all_news) < 3:
            for feed_name, feed_url in self.ALT_FEEDS.items():
                news = self._parse_feed(feed_url, feed_name)
                for item in news:
                    if item["url"] not in seen_urls:
                        seen_urls.add(item["url"])
                        all_news.append(item)

        all_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return all_news

    def _parse_feed(self, url: str, feed_name: str) -> list[dict]:
        """Tek bir RSS feed'i parse et."""
        response = self._safe_get(url)
        if not response:
            return []

        news_items = []
        try:
            feed = feedparser.parse(response.content)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")

                # Tarih
                published = datetime.now().isoformat()
                date_str = entry.get("published")
                if date_str:
                    try:
                        published = parsedate_to_datetime(date_str).isoformat()
                    except Exception:
                        pass

                # Haber tipi
                news_type = "genel"
                if "borsa" in feed_name:
                    news_type = "hisse"
                elif "ekonomi" in feed_name:
                    news_type = "makro"
                elif "dunya" in feed_name:
                    news_type = "makro"

                # Makro anahtar kelime tespiti
                text_upper = f"{title} {summary}".upper()
                macro_cats = {
                    "faiz": ["FAİZ", "FAIZ", "MERKEZ BANKASI", "TCMB"],
                    "doviz": ["DOLAR", "EURO", "DÖVİZ", "KUR"],
                    "petrol": ["PETROL", "BRENT"],
                    "altin": ["ALTIN", "GOLD"],
                }
                category = None
                for cat, keywords in macro_cats.items():
                    if any(kw in text_upper for kw in keywords):
                        category = cat
                        news_type = "makro"
                        break

                news_items.append(
                    self._make_news_item(
                        title=title,
                        url=link,
                        summary=summary,
                        published_at=published,
                        news_type=news_type,
                        category=category,
                        raw_data={"feed": feed_name},
                    )
                )
        except Exception as e:
            print(f"[{self.SOURCE_NAME}] Feed parse hatası ({feed_name}): {e}")

        return news_items
