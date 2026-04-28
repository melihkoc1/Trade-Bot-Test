"""
Tradebot V1 — Gelişmiş Haber Analizi Modülü
=============================================
Modüler haber çekme, duygu analizi ve veritabanı yönetimi.

Bileşenler:
  - fetchers/   : Kaynak bazlı haber çekiciler (Investing, KAP, Bigpara, vb.)
  - sentiment/  : BERTurk fine-tune + Gemini ensemble duygu analizi
  - database.py : SQLite haber veritabanı
  - classifier.py : Haber → hisse/sektör/makro eşleştirme
  - macro_impact.py : Makro haber → sektör etki hesaplama
  - news_manager.py : Ana orkestrasyon sınıfı
  - scheduler.py : APScheduler arka plan servisi
"""

from news.news_manager import NewsManager

__all__ = ["NewsManager"]
