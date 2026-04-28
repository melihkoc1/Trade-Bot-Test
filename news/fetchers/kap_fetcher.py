"""
Tradebot V1 — KAP (Kamuyu Aydınlatma Platformu) Fetcher
=========================================================
KAP.gov.tr'den resmi şirket bildirimlerini çeker.
Özel durum açıklamaları, finansal tablolar, temettü kararları vb.
"""

import re
from datetime import datetime
from bs4 import BeautifulSoup
from news.fetchers.base_fetcher import BaseFetcher


class KAPFetcher(BaseFetcher):
    """KAP.gov.tr resmi bildirim çekici."""

    SOURCE_NAME = "kap"
    DEFAULT_TIMEOUT = 30
    REQUEST_DELAY = 1.0  # KAP'a nazik olalım

    BASE_URL = "https://www.kap.org.tr"
    DISCLOSURES_URL = f"{BASE_URL}/tr/api/disclosures"

    # KAP disclosure endpoint'leri
    ENDPOINTS = {
        "general": f"{BASE_URL}/tr/bildirim-sorgu",
        "api_latest": f"{BASE_URL}/tr/api/disclosures?ts=0",
    }

    # Bildirim tipi ağırlıkları (strateji skoru etkisi)
    DISCLOSURE_WEIGHTS = {
        "ODA": 1.5,  # Özel Durum Açıklaması (yüksek etki)
        "FR": 1.2,  # Finansal Rapor
        "BD": 1.0,  # Bağımsız Denetim
        "HBR": 0.8,  # Haberler
        "GK": 0.7,  # Genel Kurul
        "DG": 0.5,  # Diğer
    }

    # BIST sembol → KAP kodu eşleştirmesi
    # KAP'ta semboller .IS uzantısı olmadan kullanılır
    SYMBOL_MAP = {
        "AKBNK.IS": "AKBNK",
        "ALARK.IS": "ALARK",
        "ARCLK.IS": "ARCLK",
        "ASELS.IS": "ASELS",
        "ASTOR.IS": "ASTOR",
        "BIMAS.IS": "BIMAS",
        "BRSAN.IS": "BRSAN",
        "CWENE.IS": "CWENE",
        "DOAS.IS": "DOAS",
        "EGEEN.IS": "EGEEN",
        "EKGYO.IS": "EKGYO",
        "ENJSA.IS": "ENJSA",
        "ENKAI.IS": "ENKAI",
        "EREGL.IS": "EREGL",
        "EUPWR.IS": "EUPWR",
        "FROTO.IS": "FROTO",
        "GARAN.IS": "GARAN",
        "GESAN.IS": "GESAN",
        "GUBRF.IS": "GUBRF",
        "GWIND.IS": "GWIND",
        "HALKB.IS": "HALKB",
        "HEKTS.IS": "HEKTS",
        "ISCTR.IS": "ISCTR",
        "ISGYO.IS": "ISGYO",
        "ISMEN.IS": "ISMEN",
        "KCHOL.IS": "KCHOL",
        "KONTR.IS": "KONTR",
        "KOZAA.IS": "KOZAA",
        "KOZAL.IS": "KOZAL",
        "KRDMD.IS": "KRDMD",
        "MGROS.IS": "MGROS",
        "ODAS.IS": "ODAS",
        "OYAKC.IS": "OYAKC",
        "PETKM.IS": "PETKM",
        "PGSUS.IS": "PGSUS",
        "SAHOL.IS": "SAHOL",
        "SASA.IS": "SASA",
        "SISE.IS": "SISE",
        "SMRTG.IS": "SMRTG",
        "SOKM.IS": "SOKM",
        "TAVHL.IS": "TAVHL",
        "TCELL.IS": "TCELL",
        "THYAO.IS": "THYAO",
        "TKFEN.IS": "TKFEN",
        "TOASO.IS": "TOASO",
        "TSKB.IS": "TSKB",
        "TTKOM.IS": "TTKOM",
        "TUPRS.IS": "TUPRS",
        "VAKBN.IS": "VAKBN",
        "YKBNK.IS": "YKBNK",
        "ZOREN.IS": "ZOREN",
        "CINFO.IS": "CINFO",
    }

    def __init__(self):
        super().__init__()
        # KAP'ın bot koruması için özel header'lar
        self._session.headers.update(
            {
                "Accept": "application/json, text/html",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
                "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
            }
        )

    def fetch(
        self, symbol: str = None, sector: str = None, category: str = None
    ) -> list[dict]:
        """KAP bildirimlerini çek (sembol bazlı veya genel)."""
        if symbol:
            return self._fetch_by_symbol(symbol)
        return self.fetch_all()

    def fetch_all(self) -> list[dict]:
        """Son KAP bildirimlerini çek."""
        return self._fetch_latest_disclosures()

    def _fetch_by_symbol(self, symbol: str) -> list[dict]:
        """Belirli bir sembol için KAP bildirimlerini çek."""
        kap_code = self.SYMBOL_MAP.get(symbol, symbol.replace(".IS", ""))
        full_symbol = symbol if ".IS" in symbol else f"{symbol}.IS"

        # KAP API endpoint'i ile dene
        url = f"{self.BASE_URL}/tr/api/disclosures?company={kap_code}"
        response = self._safe_get(url)

        if response:
            try:
                data = response.json()
                return self._parse_api_response(data, full_symbol)
            except Exception:
                pass

        # Fallback: HTML scraping
        return self._scrape_company_page(kap_code, full_symbol)

    def _fetch_latest_disclosures(self) -> list[dict]:
        """En son KAP bildirimlerini API üzerinden çek."""
        # Önce API'yi dene (birden fazla endpoint)
        for endpoint_key in ["api_latest"]:
            url = self.ENDPOINTS.get(endpoint_key)
            if not url:
                continue
            response = self._safe_get(url)
            if response:
                try:
                    data = response.json()
                    result = self._parse_api_response(data)
                    if result:
                        return result
                except Exception as e:
                    print(
                        f"[{self.SOURCE_NAME}] API parse hatası ({endpoint_key}): {e}"
                    )

        # Fallback 1: KAP bildirim sorgu sayfası HTML scrape
        result = self._scrape_main_page()
        if result:
            return result

        # Fallback 2: Bigpara'daki KAP haberleri sayfasını scrape et
        return self._scrape_bigpara_kap()

    def _parse_api_response(
        self, data: list | dict, default_symbol: str = None
    ) -> list[dict]:
        """KAP API JSON yanıtını standart formata dönüştür."""
        news_items = []
        items = data if isinstance(data, list) else data.get("disclosures", [])

        for item in items[:30]:  # Son 30 bildirim
            try:
                title = item.get("title", item.get("subject", ""))
                company = item.get("companyName", item.get("company", ""))
                stock_code = item.get("stockCodes", item.get("stockCode", ""))
                disc_type = item.get("disclosureType", item.get("type", "DG"))
                disc_id = item.get("disclosureIndex", item.get("id", ""))

                # Tarih
                published = item.get("publishDate", item.get("date", ""))
                if published:
                    try:
                        published = datetime.fromisoformat(
                            published.replace("Z", "+00:00")
                        ).isoformat()
                    except Exception:
                        published = datetime.now().isoformat()

                # URL oluştur
                url = f"{self.BASE_URL}/tr/bildirim/{disc_id}" if disc_id else ""

                # Sembol tespiti
                symbols = []
                if default_symbol:
                    symbols = [default_symbol]
                elif stock_code:
                    codes = (
                        stock_code.split(",")
                        if isinstance(stock_code, str)
                        else [stock_code]
                    )
                    for code in codes:
                        code = code.strip()
                        full = f"{code}.IS"
                        if full in self.SYMBOL_MAP or code in self.SYMBOL_MAP.values():
                            symbols.append(full)

                full_title = (
                    f"[KAP] {company}: {title}" if company else f"[KAP] {title}"
                )

                news_items.append(
                    self._make_news_item(
                        title=full_title,
                        url=url,
                        summary=f"KAP Bildirimi - {disc_type}: {title}",
                        published_at=published,
                        news_type="kap",
                        category=disc_type,
                        symbols=symbols,
                        raw_data={
                            "disclosure_type": disc_type,
                            "company": company,
                            "stock_code": stock_code,
                            "weight": self.DISCLOSURE_WEIGHTS.get(disc_type, 0.5),
                        },
                    )
                )
            except Exception as e:
                print(f"[{self.SOURCE_NAME}] Bildirim parse hatası: {e}")

        return news_items

    def _scrape_main_page(self) -> list[dict]:
        """KAP ana sayfasından HTML scraping ile bildirim çek."""
        response = self._safe_get(f"{self.BASE_URL}/tr/bildirim-sorgu")
        if not response:
            return []

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            return self._parse_html_disclosures(soup)
        except Exception as e:
            print(f"[{self.SOURCE_NAME}] HTML scrape hatası: {e}")
            return []

    def _scrape_company_page(self, kap_code: str, full_symbol: str) -> list[dict]:
        """Şirket sayfasından bildirimler çek."""
        url = f"{self.BASE_URL}/tr/bist-sirketler/{kap_code}"
        response = self._safe_get(url)
        if not response:
            return []

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            items = self._parse_html_disclosures(soup)
            for item in items:
                item["symbols"] = [full_symbol]
            return items
        except Exception as e:
            print(f"[{self.SOURCE_NAME}] Şirket sayfası scrape hatası: {e}")
            return []

    def _parse_html_disclosures(self, soup: BeautifulSoup) -> list[dict]:
        """BeautifulSoup ile bildirim parse et."""
        news_items = []

        # KAP bildirim tablosu
        rows = soup.select("table.table tbody tr, .disclosure-item, .notification-row")
        for row in rows[:20]:
            try:
                # Farklı HTML yapılarını dene
                title_el = row.select_one("td:nth-child(4), .title, .subject")
                company_el = row.select_one("td:nth-child(2), .company")
                date_el = row.select_one("td:nth-child(1), .date")
                link_el = row.select_one("a[href]")

                title = title_el.get_text(strip=True) if title_el else ""
                company = company_el.get_text(strip=True) if company_el else ""
                date_str = date_el.get_text(strip=True) if date_el else ""
                url = ""
                if link_el and link_el.get("href"):
                    href = link_el["href"]
                    url = href if href.startswith("http") else f"{self.BASE_URL}{href}"

                if not title:
                    continue

                # Tarih parse
                published = datetime.now().isoformat()
                if date_str:
                    for fmt in ["%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"]:
                        try:
                            published = datetime.strptime(date_str, fmt).isoformat()
                            break
                        except ValueError:
                            continue

                full_title = (
                    f"[KAP] {company}: {title}" if company else f"[KAP] {title}"
                )

                news_items.append(
                    self._make_news_item(
                        title=full_title,
                        url=url or f"kap-{hash(title)}",
                        summary=f"KAP Bildirimi: {title}",
                        published_at=published,
                        news_type="kap",
                        category="ODA",
                        raw_data={"company": company},
                    )
                )
            except Exception:
                continue

        return news_items

    def _scrape_bigpara_kap(self) -> list[dict]:
        """Bigpara'nın KAP haberleri sayfasından bildirim çek (fallback)."""
        url = "https://bigpara.hurriyet.com.tr/haberler/kap-haberleri/"
        response = self._safe_get(url)
        if not response:
            return []

        news_items = []
        try:
            soup = BeautifulSoup(response.text, "html.parser")

            # ContentLeft alanındaki haber listesi
            content = soup.select_one("div.contentLeft")
            if not content:
                content = soup

            items = content.select("div.simpleTable ul li, div.mBot20 ul li")
            # Menü li'lerini filtrele
            items = [
                i
                for i in items
                if not i.find_parent(class_="subMenu")
                and not i.find_parent(class_="menu")
                and not i.find_parent(class_="hoverMenu")
            ]

            for item in items[:20]:
                try:
                    link = item.select_one("a[href]")
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    if not title or len(title) < 15:
                        continue

                    href = link.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"https://bigpara.hurriyet.com.tr{href}"

                    # Tarih
                    date_el = item.select_one("span, time, .date")
                    published = datetime.now().isoformat()
                    if date_el:
                        date_text = date_el.get_text(strip=True)
                        if date_text and len(date_text) >= 8:
                            for fmt in [
                                "%d.%m.%Y %H:%M",
                                "%d/%m/%Y %H:%M",
                                "%d.%m.%Y",
                            ]:
                                try:
                                    published = datetime.strptime(
                                        date_text.strip(), fmt
                                    ).isoformat()
                                    break
                                except ValueError:
                                    continue

                    # Hisse sembolü tespiti (***THYAO*** formatı)
                    symbols = []
                    stock_match = re.findall(r"\*{3}(\w+)\*{3}", title)
                    for code in stock_match:
                        full = f"{code}.IS"
                        if full in self.SYMBOL_MAP or code in self.SYMBOL_MAP.values():
                            symbols.append(full)

                    news_items.append(
                        self._make_news_item(
                            title=f"[KAP] {title}",
                            url=href,
                            summary=f"KAP Bildirimi (Bigpara): {title}",
                            published_at=published,
                            news_type="kap",
                            category="ODA",
                            symbols=symbols,
                            raw_data={"via": "bigpara_kap"},
                        )
                    )
                except Exception:
                    continue

        except Exception as e:
            print(f"[{self.SOURCE_NAME}] Bigpara KAP scrape hatası: {e}")

        return news_items
