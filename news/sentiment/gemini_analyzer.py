"""
Tradebot V1 — Gemini LLM Duygu Analizi
========================================
OpenRouter API üzerinden Gemini 2.0 Flash ile haber duygu analizi.
Mevcut news_scraper.py'deki analyze_sentiment_gemini fonksiyonunun
modüler ve geliştirilmiş versiyonu.
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()


class GeminiAnalyzer:
    """Gemini LLM tabanlı haber duygu analizi."""

    MODEL = "google/gemini-2.0-flash-001"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self._last_call = 0
        self._min_delay = 0.5  # API rate limit koruması

    @property
    def available(self) -> bool:
        """API key mevcut mu?"""
        return bool(self.api_key)

    def analyze(
        self,
        title: str,
        summary: str = "",
        symbol: str = None,
        news_type: str = "genel",
    ) -> dict:
        """
        Tek bir haber için duygu analizi yap.

        Returns:
            {"score": float, "label": str, "confidence": float, "model": "gemini"}
            score: 0-100 (0=çok negatif, 50=nötr, 100=çok pozitif)
        """
        if not self.available:
            return self._default_result()

        # Rate limiting
        elapsed = time.time() - self._last_call
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_call = time.time()

        prompt = self._build_prompt(title, summary, symbol, news_type)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                self.API_URL, headers=headers, json=payload, timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                match = re.search(r"\d+", content)
                if match:
                    score = int(match.group())
                    score = max(0, min(100, score))
                    return {
                        "score": score,
                        "label": self._score_to_label(score),
                        "confidence": self._score_to_confidence(score),
                        "model": "gemini",
                    }
            else:
                print(f"[Gemini] API hatası: HTTP {response.status_code}")
        except Exception as e:
            print(f"[Gemini] İstek hatası: {e}")

        return self._default_result()

    def analyze_batch(self, items: list[dict]) -> list[dict]:
        """
        Toplu haber analizi.
        Her item: {"title": str, "summary": str, "symbol": str, "news_type": str}
        """
        results = []
        for item in items:
            result = self.analyze(
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                symbol=item.get("symbol"),
                news_type=item.get("news_type", "genel"),
            )
            results.append(result)
        return results

    def _build_prompt(
        self, title: str, summary: str, symbol: str = None, news_type: str = "genel"
    ) -> str:
        """Haber tipine göre özelleştirilmiş prompt oluştur."""
        context = ""
        if news_type == "kap":
            context = (
                "Bu bir resmi KAP (Kamuyu Aydınlatma Platformu) bildirimidir. "
                "Etkisi doğrudan ve yüksektir."
            )
        elif news_type == "sektor":
            context = (
                "Bu haber doğrudan şirket hakkında olmayabilir ancak "
                "şirketin faaliyet gösterdiği sektörü etkileyebilir."
            )
        elif news_type == "makro":
            context = (
                "Bu bir makroekonomik haber/veridir (faiz, enflasyon, döviz vb.). "
                "Tüm piyasayı ve belirli sektörleri farklı şekilde etkileyebilir."
            )

        target = f"İlgili Şirket/Konu: {symbol}" if symbol else "Genel piyasa haberi"

        return f"""Sen kıdemli bir borsa analistisin. Aşağıdaki gelişmeyi oku.
{context}

{target}
Başlık: {title}
Özet: {summary[:500]}

GÖREV: Bu haberin/bildirimin borsa/piyasa üzerindeki KISA-ORTA vadede etkisini puanla.
- 0: Çok Negatif (İflas, büyük zarar, ciddi kriz vb.)
- 25: Negatif (Zarar, ceza, olumsuz gelişme)
- 50: Nötr (Etkisi belirsiz veya rutin)
- 75: Pozitif (İyi bilanço, yeni yatırım, olumlu gelişme)
- 100: Çok Pozitif (Rekor kar, büyük ihale, bedelsiz sermaye artırımı)

SADECE 0-100 arasında bir sayı döndür. Açıklama yapma."""

    @staticmethod
    def _score_to_label(score: int) -> str:
        """Skoru 3'lü etiket sınıfına dönüştür (BERTurk ve ensemble ile uyumlu)."""
        if score >= 60:
            return "pozitif"
        elif score >= 40:
            return "notr"
        else:
            return "negatif"

    @staticmethod
    def _score_to_confidence(score: int) -> float:
        """Skor kesinliğini hesapla (50'den uzaklık = yüksek kesinlik)."""
        distance = abs(score - 50) / 50
        return round(min(1.0, 0.5 + distance * 0.5), 2)

    @staticmethod
    def _default_result() -> dict:
        """API erişilemezse varsayılan sonuç."""
        return {
            "score": 50,
            "label": "notr",
            "confidence": 0.0,
            "model": "gemini",
        }
