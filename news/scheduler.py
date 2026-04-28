"""
Tradebot V1 — Haber Zamanlayıcı (Scheduler)
=============================================
APScheduler ile arka planda otomatik haber çekme servisi.

Kullanım:
    from news.scheduler import NewsScheduler
    scheduler = NewsScheduler()
    scheduler.start()   # Arka planda başlat
    scheduler.stop()    # Durdur
"""

import threading
import time
from datetime import datetime


class NewsScheduler:
    """APScheduler ile arka plan haber çekme servisi."""

    # Varsayılan çekme aralıkları (saniye)
    INTERVALS = {
        "kap": 300,  # 5 dakika — KAP kritik, sık kontrol
        "investing": 600,  # 10 dakika
        "bigpara": 600,  # 10 dakika
        "foreks": 600,  # 10 dakika
        "tcmb": 1800,  # 30 dakika — TCMB nadir güncellenir
    }

    def __init__(self, news_manager=None):
        self._manager = news_manager
        self._running = False
        self._thread = None
        self._last_fetch = {}

    @property
    def manager(self):
        """Lazy NewsManager yükleme."""
        if self._manager is None:
            from news.news_manager import get_manager

            self._manager = get_manager()
        return self._manager

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Arka plan zamanlayıcısını başlat."""
        if self._running:
            print("[Scheduler] Zaten çalışıyor.")
            return

        # APScheduler ile dene, yoksa basit thread kullan
        try:
            self._start_apscheduler()
        except ImportError:
            print(
                "[Scheduler] APScheduler bulunamadı, basit thread scheduler kullanılıyor."
            )
            self._start_simple()

    def stop(self):
        """Zamanlayıcıyı durdur."""
        self._running = False
        if hasattr(self, "_apscheduler"):
            try:
                self._apscheduler.shutdown(wait=False)
            except Exception:
                pass
        print("[Scheduler] Durduruldu.")

    def _start_apscheduler(self):
        """APScheduler ile başlat."""
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler()

        for source, interval in self.INTERVALS.items():
            scheduler.add_job(
                self._fetch_source,
                "interval",
                seconds=interval,
                args=[source],
                id=f"news_{source}",
                name=f"Haber çekme: {source}",
                max_instances=1,
                coalesce=True,
            )

        # Temizlik görevi (günde bir)
        scheduler.add_job(
            self._cleanup,
            "cron",
            hour=3,
            minute=0,
            id="news_cleanup",
            name="Eski haber temizliği",
        )

        scheduler.start()
        self._apscheduler = scheduler
        self._running = True
        print(
            f"[Scheduler] APScheduler başlatıldı — {len(self.INTERVALS)} kaynak aktif"
        )

    def _start_simple(self):
        """Basit thread-based scheduler."""
        self._running = True
        self._thread = threading.Thread(target=self._simple_loop, daemon=True)
        self._thread.start()
        print(
            f"[Scheduler] Basit scheduler başlatıldı — {len(self.INTERVALS)} kaynak aktif"
        )

    def _simple_loop(self):
        """Basit döngü tabanlı zamanlayıcı."""
        while self._running:
            now = time.time()
            for source, interval in self.INTERVALS.items():
                last = self._last_fetch.get(source, 0)
                if now - last >= interval:
                    self._fetch_source(source)
                    self._last_fetch[source] = now
            time.sleep(30)  # 30 saniye aralıklarla kontrol

    def _fetch_source(self, source: str):
        """Tek bir kaynaktan haber çek ve veritabanına kaydet."""
        try:
            fetcher = self.manager.fetchers.get(source)
            if not fetcher:
                return

            news = fetcher.fetch_all()
            if not news:
                return

            # Sınıflandır
            news = self.manager.classifier.classify_batch(news)

            # Veritabanına kaydet
            if self.manager.db:
                inserted = self.manager.db.insert_news_batch(news)
                if inserted > 0:
                    print(
                        f"[Scheduler] {source}: {inserted} yeni haber eklendi "
                        f"({datetime.now().strftime('%H:%M')})"
                    )

            self._last_fetch[source] = time.time()

        except Exception as e:
            print(f"[Scheduler] {source} fetch hatası: {e}")

    def _cleanup(self):
        """Eski haberleri temizle."""
        if self.manager.db:
            deleted = self.manager.db.cleanup_old_news(days=30)
            if deleted > 0:
                print(f"[Scheduler] {deleted} eski haber silindi.")

    def fetch_now(self, source: str = None):
        """Manuel olarak şimdi çek."""
        if source:
            self._fetch_source(source)
        else:
            for src in self.INTERVALS:
                self._fetch_source(src)

    def get_status(self) -> dict:
        """Zamanlayıcı durum bilgisi."""
        return {
            "running": self._running,
            "sources": list(self.INTERVALS.keys()),
            "intervals": self.INTERVALS,
            "last_fetch": {
                src: datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                for src, ts in self._last_fetch.items()
            },
        }
