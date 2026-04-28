"""
Tradebot V1 — Haber Veritabanı (SQLite)
========================================
Haberlerin, duygu analizi sonuçlarının ve makro etkilerin
kalıcı olarak saklanması ve sorgulanması.
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "news_store.db"
)


class NewsDatabase:
    """SQLite tabanlı haber veritabanı yöneticisi."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Thread-safe bağlantı context manager."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Veritabanı tablolarını oluştur."""
        with self._get_conn() as conn:
            conn.executescript("""
                -- Ana haber tablosu
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    url TEXT UNIQUE,
                    published_at TEXT,
                    fetched_at TEXT NOT NULL,
                    news_type TEXT DEFAULT 'genel',
                    category TEXT,
                    raw_data TEXT
                );

                -- Haber-Hisse eşleştirme tablosu
                CREATE TABLE IF NOT EXISTS news_symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    match_type TEXT DEFAULT 'direct',
                    FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE,
                    UNIQUE(news_id, symbol)
                );

                -- Duygu analizi sonuçları
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    label TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL,
                    FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE
                );

                -- Makro etki tablosu
                CREATE TABLE IF NOT EXISTS macro_impact (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id INTEGER NOT NULL,
                    sector TEXT NOT NULL,
                    impact_direction TEXT NOT NULL,
                    impact_score REAL NOT NULL,
                    macro_type TEXT,
                    FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE
                );

                -- İndeksler (performans için)
                CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);
                CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at);
                CREATE INDEX IF NOT EXISTS idx_news_fetched ON news(fetched_at);
                CREATE INDEX IF NOT EXISTS idx_news_type ON news(news_type);
                CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON news_symbols(symbol);
                CREATE INDEX IF NOT EXISTS idx_symbols_news ON news_symbols(news_id);
                CREATE INDEX IF NOT EXISTS idx_sentiment_news ON sentiment_scores(news_id);
                CREATE INDEX IF NOT EXISTS idx_sentiment_model ON sentiment_scores(model);
                CREATE INDEX IF NOT EXISTS idx_macro_sector ON macro_impact(sector);
            """)

    # ──────────────────────────────────────────────────
    # CRUD — Haber
    # ──────────────────────────────────────────────────

    def insert_news(
        self,
        source: str,
        title: str,
        url: str,
        summary: str = None,
        published_at: str = None,
        news_type: str = "genel",
        category: str = None,
        raw_data: dict = None,
    ) -> int | None:
        """Yeni bir haber ekle. URL zaten varsa None döndürür (duplicate skip)."""
        now = datetime.now().isoformat()
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO news 
                    (source, title, summary, url, published_at, fetched_at, news_type, category, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        source,
                        title,
                        summary,
                        url,
                        published_at,
                        now,
                        news_type,
                        category,
                        json.dumps(raw_data, ensure_ascii=False) if raw_data else None,
                    ),
                )
                if cursor.rowcount == 0:
                    return None  # Duplicate URL
                return cursor.lastrowid
        except Exception as e:
            print(f"[NewsDB] Haber ekleme hatası: {e}")
            return None

    def insert_news_batch(self, news_list: list[dict]) -> int:
        """Toplu haber ekleme. Dönen: eklenen sayısı."""
        now = datetime.now().isoformat()
        inserted = 0
        with self._get_conn() as conn:
            for n in news_list:
                try:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO news 
                        (source, title, summary, url, published_at, fetched_at, news_type, category, raw_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            n.get("source", "unknown"),
                            n["title"],
                            n.get("summary"),
                            n.get("url", ""),
                            n.get("published_at"),
                            now,
                            n.get("news_type", "genel"),
                            n.get("category"),
                            json.dumps(n.get("raw_data"), ensure_ascii=False)
                            if n.get("raw_data")
                            else None,
                        ),
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                        # Eğer news_id dönmüşse ve symbols varsa, eşleştir
                        news_id = cursor.lastrowid
                        for sym_info in n.get("symbols", []):
                            if isinstance(sym_info, str):
                                sym_info = {"symbol": sym_info}
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO news_symbols (news_id, symbol, relevance_score, match_type)
                                VALUES (?, ?, ?, ?)
                            """,
                                (
                                    news_id,
                                    sym_info["symbol"],
                                    sym_info.get("relevance_score", 1.0),
                                    sym_info.get("match_type", "direct"),
                                ),
                            )
                except Exception as e:
                    print(f"[NewsDB] Batch ekleme hatası: {e}")
        return inserted

    def get_news_by_symbol(
        self, symbol: str, hours: int = 24, limit: int = 20
    ) -> list[dict]:
        """Belirli bir sembol için son N saatteki haberleri getir."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT n.*, ns.relevance_score, ns.match_type
                FROM news n
                JOIN news_symbols ns ON n.id = ns.news_id
                WHERE ns.symbol = ? AND n.fetched_at >= ?
                ORDER BY n.published_at DESC
                LIMIT ?
            """,
                (symbol, since, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_news_by_sector(
        self, sector: str, hours: int = 24, limit: int = 20
    ) -> list[dict]:
        """Sektör bazlı haberleri getir (news_type alanından)."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM news 
                WHERE category = ? AND fetched_at >= ?
                ORDER BY published_at DESC
                LIMIT ?
            """,
                (sector, since, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_news_by_type(
        self, news_type: str, hours: int = 24, limit: int = 20
    ) -> list[dict]:
        """Haber tipine göre (kap, makro, hisse, sektor, genel) getir."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM news 
                WHERE news_type = ? AND fetched_at >= ?
                ORDER BY published_at DESC
                LIMIT ?
            """,
                (news_type, since, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_news(self, hours: int = 24, limit: int = 50) -> list[dict]:
        """Son N saatteki tüm haberleri getir."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM news 
                WHERE fetched_at >= ?
                ORDER BY published_at DESC
                LIMIT ?
            """,
                (since, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_unanalyzed_news(self, model: str = None, limit: int = 50) -> list[dict]:
        """Henüz duygu analizi yapılmamış haberleri getir."""
        with self._get_conn() as conn:
            if model:
                rows = conn.execute(
                    """
                    SELECT n.* FROM news n
                    LEFT JOIN sentiment_scores s ON n.id = s.news_id AND s.model = ?
                    WHERE s.id IS NULL
                    ORDER BY n.published_at DESC
                    LIMIT ?
                """,
                    (model, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT n.* FROM news n
                    LEFT JOIN sentiment_scores s ON n.id = s.news_id
                    WHERE s.id IS NULL
                    ORDER BY n.published_at DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────
    # CRUD — Sembol Eşleştirme
    # ──────────────────────────────────────────────────

    def link_news_to_symbol(
        self,
        news_id: int,
        symbol: str,
        relevance_score: float = 1.0,
        match_type: str = "direct",
    ) -> bool:
        """Haberi bir sembolle eşleştir."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO news_symbols (news_id, symbol, relevance_score, match_type)
                    VALUES (?, ?, ?, ?)
                """,
                    (news_id, symbol, relevance_score, match_type),
                )
                return True
        except Exception as e:
            print(f"[NewsDB] Sembol eşleştirme hatası: {e}")
            return False

    # ──────────────────────────────────────────────────
    # CRUD — Duygu Analizi
    # ──────────────────────────────────────────────────

    def insert_sentiment(
        self, news_id: int, model: str, score: float, confidence: float, label: str
    ) -> int | None:
        """Duygu analizi sonucu ekle."""
        now = datetime.now().isoformat()
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO sentiment_scores (news_id, model, score, confidence, label, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (news_id, model, score, confidence, label, now),
                )
                return cursor.lastrowid
        except Exception as e:
            print(f"[NewsDB] Sentiment ekleme hatası: {e}")
            return None

    def get_sentiment_for_symbol(
        self, symbol: str, model: str = "ensemble", hours: int = 24
    ) -> list[dict]:
        """Bir sembol için duygu analizi sonuçlarını getir."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT s.*, n.title, n.source, n.news_type, ns.relevance_score
                FROM sentiment_scores s
                JOIN news n ON s.news_id = n.id
                JOIN news_symbols ns ON n.id = ns.news_id
                WHERE ns.symbol = ? AND s.model = ? AND s.analyzed_at >= ?
                ORDER BY n.published_at DESC
            """,
                (symbol, model, since),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_avg_sentiment(
        self, symbol: str, model: str = "ensemble", hours: int = 24
    ) -> dict:
        """Bir sembol için ortalama duygu skoru hesapla."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT 
                    AVG(s.score) as avg_score,
                    AVG(s.confidence) as avg_confidence,
                    COUNT(*) as news_count,
                    SUM(CASE WHEN s.label IN ('pozitif', 'cok_pozitif') THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN s.label IN ('negatif', 'cok_negatif') THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN s.label = 'notr' THEN 1 ELSE 0 END) as neutral_count
                FROM sentiment_scores s
                JOIN news n ON s.news_id = n.id
                JOIN news_symbols ns ON n.id = ns.news_id
                WHERE ns.symbol = ? AND s.model = ? AND s.analyzed_at >= ?
            """,
                (symbol, model, since),
            ).fetchone()
            return dict(row) if row else {}

    # ──────────────────────────────────────────────────
    # CRUD — Makro Etki
    # ──────────────────────────────────────────────────

    def insert_macro_impact(
        self,
        news_id: int,
        sector: str,
        impact_direction: str,
        impact_score: float,
        macro_type: str = None,
    ) -> int | None:
        """Makro etki kaydı ekle."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO macro_impact (news_id, sector, impact_direction, impact_score, macro_type)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (news_id, sector, impact_direction, impact_score, macro_type),
                )
                return cursor.lastrowid
        except Exception as e:
            print(f"[NewsDB] Makro etki ekleme hatası: {e}")
            return None

    def get_macro_impact_for_sector(self, sector: str, hours: int = 24) -> list[dict]:
        """Sektör için makro etkileri getir."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT m.*, n.title, n.source
                FROM macro_impact m
                JOIN news n ON m.news_id = n.id
                WHERE m.sector = ? AND n.fetched_at >= ?
                ORDER BY n.published_at DESC
            """,
                (sector, since),
            ).fetchall()
            return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────
    # Eğitim Verisi Dışa Aktarma
    # ──────────────────────────────────────────────────

    def export_for_training(
        self, output_path: str = None, min_confidence: float = 0.0
    ) -> int:
        """Etiketlenmiş haberleri BERTurk eğitimi için CSV olarak dışa aktar."""
        import csv

        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "news_training",
                "labeled_dataset.csv",
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT n.title, n.summary, s.label, s.score, s.confidence, s.model, n.source
                FROM sentiment_scores s
                JOIN news n ON s.news_id = n.id
                WHERE s.confidence >= ?
                ORDER BY s.analyzed_at DESC
            """,
                (min_confidence,),
            ).fetchall()

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["title", "summary", "label", "score", "confidence", "model", "source"]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["title"],
                        r["summary"],
                        r["label"],
                        r["score"],
                        r["confidence"],
                        r["model"],
                        r["source"],
                    ]
                )

        print(f"[NewsDB] {len(rows)} etiketli haber dışa aktarıldı: {output_path}")
        return len(rows)

    # ──────────────────────────────────────────────────
    # Temizlik
    # ──────────────────────────────────────────────────

    def cleanup_old_news(self, days: int = 30) -> int:
        """Belirtilen günden eski haberleri sil."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM news WHERE fetched_at < ?", (cutoff,))
            return cursor.rowcount

    def get_stats(self) -> dict:
        """Veritabanı istatistikleri."""
        with self._get_conn() as conn:
            stats = {}
            stats["total_news"] = conn.execute("SELECT COUNT(*) FROM news").fetchone()[
                0
            ]
            stats["total_sentiments"] = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores"
            ).fetchone()[0]
            stats["total_symbols_linked"] = conn.execute(
                "SELECT COUNT(*) FROM news_symbols"
            ).fetchone()[0]
            stats["total_macro_impacts"] = conn.execute(
                "SELECT COUNT(*) FROM macro_impact"
            ).fetchone()[0]

            # Kaynak bazlı dağılım
            rows = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM news GROUP BY source"
            ).fetchall()
            stats["by_source"] = {r["source"]: r["cnt"] for r in rows}

            # Tip bazlı dağılım
            rows = conn.execute(
                "SELECT news_type, COUNT(*) as cnt FROM news GROUP BY news_type"
            ).fetchall()
            stats["by_type"] = {r["news_type"]: r["cnt"] for r in rows}

            # Son 24 saat
            since_24h = (datetime.now() - timedelta(hours=24)).isoformat()
            stats["last_24h"] = conn.execute(
                "SELECT COUNT(*) FROM news WHERE fetched_at >= ?", (since_24h,)
            ).fetchone()[0]

            return stats
