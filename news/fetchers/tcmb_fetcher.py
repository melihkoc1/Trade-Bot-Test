"""
Tradebot V1 — TCMB Haber Fetcher
=================================
TCMB (Türkiye Cumhuriyet Merkez Bankası) duyuru ve haberlerini çeker.
Faiz kararları, enflasyon verileri, para politikası kararları.
"""

from datetime import datetime
from bs4 import BeautifulSoup
from news.fetchers.base_fetcher import BaseFetcher


class TCMBFetcher(BaseFetcher):
    """TCMB duyuru ve haber çekici."""

    SOURCE_NAME = "tcmb"
    DEFAULT_TIMEOUT = 20
    REQUEST_DELAY = 1.0

    BASE_URL = "https://www.tcmb.gov.tr"

    PAGES = {
        "anasayfa": f"{BASE_URL}",
    }

    # TCMB haberleri hangi sektörleri etkiler
    IMPACT_MAP = {
        "faiz": {
            "Banka": -0.8,  # Faiz artışı bankaları genelde olumsuz (kısa vade)
            "GYO": -0.6,  # Gayrimenkul de olumsuz
            "Holding": -0.4,
        },
        "enflasyon": {
            "Perakende": -0.5,
            "Banka": -0.3,
        },
        "kur": {
            "Enerji": 0.5,  # Dolar bazlı ihracatçılar olumlu
            "Madencilik": 0.5,
            "Ulasim": 0.3,
            "Banka": -0.6,
        },
    }

    # Kritik anahtar kelimeler
    CRITICAL_KEYWORDS = [
        "FAİZ KARARI",
        "FAIZ KARARI",
        "POLİTİKA FAİZİ",
        "ENFLASYON RAPORU",
        "PARA POLİTİKASI",
        "ZORUNLU KARŞILIK",
        "DÖVİZ MÜDAHALESİ",
        "LİKİDİTE YÖNETİMİ",
        "FİYAT İSTİKRARI",
    ]

    def fetch(
        self, symbol: str = None, sector: str = None, category: str = None
    ) -> list[dict]:
        """TCMB haberlerini çek. Sembol filtresi burada pek anlamlı değil
        çünkü TCMB haberleri makro düzeyde tüm piyasayı etkiler."""
        return self.fetch_all()

    def fetch_all(self) -> list[dict]:
        """TCMB duyurularını çek."""
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

    def _scrape_page(self, url: str, page_type: str) -> list[dict]:
        """TCMB sayfasını scrape et."""
        response = self._safe_get(url)
        if not response:
            return []

        news_items = []
        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # TCMB ana sayfa: tüm anlamlı linkleri topla
            # TCMB'de haberler a etiketleri olarak listeleniyor
            all_links = soup.select("a[href]")

            seen_titles = set()
            for link_el in all_links:
                try:
                    title = link_el.get_text(strip=True)
                    if not title or len(title) < 15:
                        continue

                    # Sadece finans/ekonomi ile ilgili haberleri al
                    title_upper = title.upper()
                    is_relevant = any(
                        kw in title_upper
                        for kw in [
                            "FAİZ",
                            "FAIZ",
                            "ENFLASYON",
                            "BASIN DUYURUSU",
                            "PARA POLİTİKASI",
                            "LİKİDİTE",
                            "DÖVİZ",
                            "ÖDEMELER DENGESİ",
                            "ÖDEMELER DENGESI",
                            "ZORUNLU KARŞILIK",
                            "REZERV",
                            "FİYAT GELİŞMELERİ",
                            "FİYAT İSTİKRARI",
                            "BİLGİLENDİRME TOPLANTISI",
                            "PPK",
                            "FİNANSAL İSTİKRAR",
                            "PARA VE BANKA",
                            "MENKUL KIYMET",
                            "STOK FAİZ",
                            "EFEKTİF DÖVİZ",
                        ]
                    )
                    if not is_relevant:
                        continue

                    # Menü / navigasyon linkleri, müze, HR vs. atla
                    if any(
                        skip in title_upper
                        for skip in [
                            "TÜM BASIN",
                            "TÜM KONUŞMALAR",
                            "TÜM VERİLER",
                            "TÜM YAYINLAR",
                            "TÜM PPK",
                            "TÜMÜNÜ",
                            "SANAL MÜZE",
                            "İNSAN KAYNAKLARI",
                            "KAMU HİZMETLERİ",
                            "KULLANıM ŞARTLARI",
                            "ERİŞİLEBİLİRLİK",
                            "HERKES İÇİN EKONOMİ",
                            "ŞİKAYET YÖNETİM",
                            "KVKK",
                        ]
                    ):
                        continue

                    # Menü linkleri ve genel navigasyonu atla
                    if any(
                        skip in title_upper
                        for skip in [
                            "TÜM BASIN",
                            "TÜM KONUŞMALAR",
                            "TÜM VERİLER",
                            "TÜM YAYINLAR",
                            "TÜM PPK",
                            "TÜMÜNÜ",
                        ]
                    ):
                        continue

                    # Tekrar kontrolü
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    href = link_el.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"{self.BASE_URL}{href}"

                    # Tarih: parent veya sibling'den al
                    published = datetime.now().isoformat()
                    parent = link_el.parent
                    if parent:
                        date_el = parent.select_one("span, time, .date")
                        if date_el:
                            date_text = date_el.get_text(strip=True)
                            if date_text and len(date_text) >= 4:
                                published = self._parse_date(date_text)

                    # Kritik haber tespiti
                    is_critical = any(
                        kw in title_upper for kw in self.CRITICAL_KEYWORDS
                    )

                    # Makro kategorisini tespit et
                    category = self._detect_category(title)

                    news_items.append(
                        self._make_news_item(
                            title=f"[TCMB] {title}",
                            url=href or f"tcmb-{hash(title)}",
                            summary=f"TCMB {page_type}: {title}",
                            published_at=published,
                            news_type="makro",
                            category=category or "tcmb_genel",
                            raw_data={
                                "page_type": page_type,
                                "is_critical": is_critical,
                                "impact_map": self.IMPACT_MAP.get(category, {}),
                            },
                        )
                    )
                except Exception:
                    continue

        except Exception as e:
            print(f"[{self.SOURCE_NAME}] Scrape hatası ({page_type}): {e}")

        return news_items

    def _detect_category(self, title: str) -> str | None:
        """Haber kategorisini başlıktan tespit et."""
        title_upper = title.upper()
        if any(kw in title_upper for kw in ["FAİZ", "FAIZ", "POLİTİKA FAİZİ"]):
            return "faiz"
        elif any(kw in title_upper for kw in ["ENFLASYON", "TÜFE", "TUFE", "FİYAT"]):
            return "enflasyon"
        elif any(kw in title_upper for kw in ["DÖVİZ", "DOVIZ", "KUR", "DOLAR"]):
            return "kur"
        elif any(kw in title_upper for kw in ["KARŞILIK", "LİKİDİTE", "ZORUNLU"]):
            return "likidite"
        return None

    def _parse_date(self, date_str: str) -> str:
        """Tarih parse."""
        if not date_str:
            return datetime.now().isoformat()

        for fmt in [
            "%d.%m.%Y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%d.%m.%Y %H:%M",
            "%d %B %Y",
        ]:
            try:
                return datetime.strptime(date_str.strip(), fmt).isoformat()
            except ValueError:
                continue
        return datetime.now().isoformat()
