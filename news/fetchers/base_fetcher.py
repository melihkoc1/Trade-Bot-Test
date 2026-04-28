"""
Tradebot V1 — Temel Haber Fetcher Soyut Sınıfı
================================================
Tüm haber kaynağı fetcher'ları bu sınıftan türetilir.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import requests
import time


class BaseFetcher(ABC):
    """Tüm haber kaynaklarının uygulaması gereken temel arayüz."""

    # Alt sınıflar tarafından override edilmeli
    SOURCE_NAME: str = "unknown"
    DEFAULT_TIMEOUT: int = 15
    REQUEST_DELAY: float = 0.3  # İstekler arası bekleme (rate limiting)

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    def __init__(self):
        self._last_request_time = 0
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

    def _rate_limit(self):
        """İstekler arası minimum bekleme süresi uygula."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _safe_get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Hata yönetimli HTTP GET isteği."""
        self._rate_limit()
        try:
            kwargs.setdefault("timeout", self.DEFAULT_TIMEOUT)
            response = self._session.get(url, **kwargs)
            if response.status_code == 200:
                return response
            else:
                print(f"[{self.SOURCE_NAME}] HTTP {response.status_code}: {url}")
                return None
        except requests.exceptions.Timeout:
            print(f"[{self.SOURCE_NAME}] Timeout: {url}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"[{self.SOURCE_NAME}] Bağlantı hatası: {url}")
            return None
        except Exception as e:
            print(f"[{self.SOURCE_NAME}] İstek hatası: {e}")
            return None

    @abstractmethod
    def fetch(
        self, symbol: str = None, sector: str = None, category: str = None
    ) -> list[dict]:
        """
        Haberleri çek ve standart formata dönüştür.

        Her bir haber dict'i şu formatta olmalı:
        {
            "source": str,          # Kaynak adı (ör: "investing", "kap")
            "title": str,           # Haber başlığı
            "summary": str,         # Haber özeti
            "url": str,             # Haber URL'i (unique identifier)
            "published_at": str,    # Yayın tarihi (ISO format)
            "news_type": str,       # "hisse", "sektor", "makro", "genel", "kap"
            "category": str,        # Alt kategori
            "symbols": list[str],   # İlişkili semboller (opsiyonel)
            "raw_data": dict,       # Ham veri (opsiyonel)
        }
        """
        pass

    @abstractmethod
    def fetch_all(self) -> list[dict]:
        """Kaynaktaki tüm güncel haberleri çek (filtresiz)."""
        pass

    def _make_news_item(
        self,
        title: str,
        url: str,
        summary: str = "",
        published_at: str = None,
        news_type: str = "genel",
        category: str = None,
        symbols: list = None,
        raw_data: dict = None,
    ) -> dict:
        """Standart haber dict'i oluştur."""
        return {
            "source": self.SOURCE_NAME,
            "title": title.strip() if title else "",
            "summary": (summary.strip()[:500] if summary else ""),
            "url": url.strip() if url else "",
            "published_at": published_at or datetime.now().isoformat(),
            "news_type": news_type,
            "category": category,
            "symbols": symbols or [],
            "raw_data": raw_data,
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} source={self.SOURCE_NAME}>"
