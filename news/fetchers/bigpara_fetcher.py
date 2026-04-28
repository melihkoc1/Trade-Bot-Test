"""
Tradebot V1 — Bigpara Haber Fetcher
====================================
Bigpara.hurriyet.com.tr'den Türk borsası haberlerini çeker.
RSS ve web scraping kombinasyonu.
"""

from datetime import datetime
from bs4 import BeautifulSoup
from news.fetchers.base_fetcher import BaseFetcher


class BigparaFetcher(BaseFetcher):
    """Bigpara haber çekici."""

    SOURCE_NAME = "bigpara"
    REQUEST_DELAY = 0.5

    BASE_URL = "https://bigpara.hurriyet.com.tr"

    # Bigpara haber sayfası URL'leri (Nisan 2026 güncel)
    PAGES = {
        "borsa": f"{BASE_URL}/borsa/haber/",
        "ekonomi": f"{BASE_URL}/haberler/ekonomi-haberleri/",
        "piyasa": f"{BASE_URL}/haberler/piyasa-haberleri/",
    }

    def fetch(
        self, symbol: str = None, sector: str = None, category: str = None
    ) -> list[dict]:
        """Bigpara haberlerini çek."""
        all_news = self.fetch_all()
        if not symbol:
            return all_news

        # Sembol filtresi
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
        """Bigpara'dan tüm haber kategorilerini çek."""
        all_news = []
        seen_urls = set()

        for page_name, page_url in self.PAGES.items():
            news = self._scrape_page(page_url, page_name)
            for item in news:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_news.append(item)

        all_news.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return all_news

    def _scrape_page(self, url: str, category: str) -> list[dict]:
        """Bigpara haber sayfasını scrape et."""
        response = self._safe_get(url)
        if not response:
            return []

        news_items = []
        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # Bigpara haber listesi yapısı
            # ContentLeft alanı içinde: div.mBot20 veya simpleTable içindeki listeler
            # Menü (ul.subMenu, ul.menu) hariç tutulmalı
            content_area = soup.select_one("div.contentLeft")
            if not content_area:
                content_area = soup

            articles = content_area.select(
                "div.simpleTable ul li, div.mBot20 ul li, div.mBot30 ul li"
            )

            # subMenu ve navMenu li'lerini filtrele
            articles = [
                a
                for a in articles
                if not a.find_parent(class_="subMenu")
                and not a.find_parent(class_="menu")
                and not a.find_parent(class_="navMenu")
                and not a.find_parent(class_="ddMenu")
                and not a.find_parent(class_="hoverMenu")
            ]

            # Eğer hala bulamadıysak, geniş arama yap
            if not articles:
                articles = content_area.select("article, .news-item, .haber-item")

            for article in articles[:20]:
                try:
                    # Başlık + Link (genelde a > h2/h3 veya doğrudan a)
                    link = article.select_one("a[href]")
                    if not link:
                        continue

                    # Başlık: a içindeki text veya h2/h3
                    title_el = article.select_one(
                        "h2, h3, h4, .title, .baslik, a.news-title"
                    )
                    if title_el:
                        title = title_el.get_text(strip=True)
                    else:
                        title = link.get_text(strip=True)

                    if not title or len(title) < 10:
                        continue

                    # URL
                    href = link.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"{self.BASE_URL}{href}"
                    if not href:
                        continue

                    # Kategori/menü sayfalarını atla
                    if any(
                        skip in href
                        for skip in [
                            "/kobi/",
                            "/analiz/",
                            "/video/",
                            "/bigpara-yazarlari/",
                            "/portfoy/",
                            "/yorumlari",
                            "/en-cok-okunan",
                            "/sondakika-haberleri/",
                            "/tumu/",
                            "javascript:",
                            "/araci-kurum-",
                            "/piyasa-takvimi",
                            "/enflasyon-verileri",
                        ]
                    ):
                        continue

                    # Özet
                    summary_el = article.select_one(
                        "p, .summary, .ozet, .spot, .description"
                    )
                    summary = summary_el.get_text(strip=True) if summary_el else ""

                    # Tarih
                    date_el = article.select_one(
                        "time, .date, .tarih, .time, span.date, span"
                    )
                    published = datetime.now().isoformat()
                    if date_el:
                        date_text = date_el.get(
                            "datetime", date_el.get_text(strip=True)
                        )
                        if date_text and len(date_text) >= 8:
                            published = self._parse_date(date_text)

                    # Haber tipi belirle
                    news_type = "genel"
                    if category == "borsa":
                        news_type = "hisse"
                    elif category == "piyasa":
                        news_type = "makro"

                    news_items.append(
                        self._make_news_item(
                            title=title,
                            url=href,
                            summary=summary,
                            published_at=published,
                            news_type=news_type,
                            category=category,
                            raw_data={"page": category},
                        )
                    )
                except Exception:
                    continue

        except Exception as e:
            print(f"[{self.SOURCE_NAME}] Scrape hatası ({category}): {e}")

        return news_items

    def _parse_date(self, date_str: str) -> str:
        """Tarih string'ini parse et."""
        if not date_str:
            return datetime.now().isoformat()

        for fmt in [
            "%d.%m.%Y %H:%M",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%d %B %Y %H:%M",
            "%d.%m.%Y",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(date_str.strip(), fmt).isoformat()
            except ValueError:
                continue

        return datetime.now().isoformat()
