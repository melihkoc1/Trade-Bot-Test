"""
Tradebot V1 — Ensemble Duygu Analizi
======================================
BERTurk fine-tune + Gemini LLM ensemble duygu analizi.
İki modelin tahminlerini ağırlıklı olarak birleştirir.

Strateji:
  - BERTurk mevcut ve çalışıyorsa: %60 BERTurk + %40 Gemini (yerel model öncelikli)
  - Sadece BERTurk: %100 BERTurk
  - Sadece Gemini: %100 Gemini
  - Hiçbiri yoksa: Eski LSTM fallback
"""

from news.sentiment.gemini_analyzer import GeminiAnalyzer
from news.sentiment.berturk_model import BERTurkSentiment


class SentimentEnsemble:
    """Ensemble duygu analizi motoru."""

    # Ağırlıklar
    WEIGHT_BERTURK = 0.6
    WEIGHT_GEMINI = 0.4

    def __init__(self):
        self.gemini = GeminiAnalyzer()
        self.berturk = BERTurkSentiment()
        self._lstm_available = None

    @property
    def engines_status(self) -> dict:
        """Mevcut motor durumları."""
        return {
            "gemini": self.gemini.available,
            "berturk": self.berturk.available,
            "lstm": self._check_lstm(),
        }

    def analyze(
        self,
        title: str,
        summary: str = "",
        symbol: str = None,
        news_type: str = "genel",
    ) -> dict:
        """
        Ensemble duygu analizi.

        Returns:
            {
                "score": float,         # 0-100 (normalleştirilmiş)
                "normalized_score": float,  # -50 ile +50 (strategy uyumlu)
                "label": str,           # 5'li sınıf etiketi
                "confidence": float,    # 0-1 güven skoru
                "model": str,           # Kullanılan model(ler)
                "details": dict,        # Alt model sonuçları
            }
        """
        results = {}
        weights = {}

        # BERTurk denemesi
        if self.berturk.available:
            text = f"{title} {summary}".strip()
            berturk_result = self.berturk.predict(text)
            if berturk_result["confidence"] > 0:
                results["berturk"] = berturk_result
                weights["berturk"] = self.WEIGHT_BERTURK

        # Gemini denemesi
        if self.gemini.available:
            gemini_result = self.gemini.analyze(title, summary, symbol, news_type)
            if gemini_result["confidence"] > 0:
                results["gemini"] = gemini_result
                weights["gemini"] = self.WEIGHT_GEMINI

        # Hiçbiri yoksa LSTM fallback
        if not results:
            lstm_result = self._lstm_fallback(title)
            if lstm_result:
                return lstm_result

            # Son çare: nötr döndür
            return self._neutral_result()

        # Ensemble hesaplama
        return self._combine_results(results, weights, news_type)

    def analyze_batch(self, items: list[dict]) -> list[dict]:
        """
        Toplu ensemble analiz.
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

    def _combine_results(self, results: dict, weights: dict, news_type: str) -> dict:
        """Alt model sonuçlarını ağırlıklı birleştir."""
        # Ağırlıkları normalize et
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # Ağırlıklı skor
        weighted_score = sum(
            results[model]["score"] * norm_weights[model] for model in results
        )

        # Ağırlıklı güven
        weighted_confidence = sum(
            results[model]["confidence"] * norm_weights[model] for model in results
        )

        # KAP haberleri daha yüksek ağırlık alır
        if news_type == "kap":
            # Nötr'den uzaklığı artır
            weighted_score = 50 + (weighted_score - 50) * 1.3
            weighted_score = max(0, min(100, weighted_score))

        score = round(weighted_score, 1)
        label = self._score_to_label(score)
        model_names = "+".join(results.keys())

        return {
            "score": score,
            "normalized_score": round(score - 50, 1),
            "label": label,
            "confidence": round(weighted_confidence, 3),
            "model": f"ensemble({model_names})",
            "details": results,
        }

    def _lstm_fallback(self, title: str) -> dict | None:
        """Eski LSTM modelini fallback olarak kullan."""
        if not self._check_lstm():
            return None

        try:
            from sentiment_lstm import get_sentiment

            label, conf = get_sentiment(title)
            label_to_score = {"POZITIF": 75, "NOTR": 50, "NEGATIF": 25}
            base_score = label_to_score.get(label, 50)
            score = 50 + (base_score - 50) * conf

            label_map = {"POZITIF": "pozitif", "NOTR": "notr", "NEGATIF": "negatif"}

            return {
                "score": round(score, 1),
                "normalized_score": round(score - 50, 1),
                "label": label_map.get(label, "notr"),
                "confidence": round(conf, 3),
                "model": "lstm_fallback",
                "details": {"lstm": {"label": label, "confidence": conf}},
            }
        except Exception:
            return None

    def _check_lstm(self) -> bool:
        """LSTM modeli mevcut mu?"""
        if self._lstm_available is not None:
            return self._lstm_available
        try:
            import os

            self._lstm_available = os.path.exists("models/sentiment_lstm.pt")
        except Exception:
            self._lstm_available = False
        return self._lstm_available

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Skoru 3'lü etikete dönüştür."""
        if score >= 60:
            return "pozitif"
        elif score >= 40:
            return "notr"
        else:
            return "negatif"

    @staticmethod
    def _neutral_result() -> dict:
        """Varsayılan nötr sonuç."""
        return {
            "score": 50,
            "normalized_score": 0,
            "label": "notr",
            "confidence": 0.0,
            "model": "none",
            "details": {},
        }
