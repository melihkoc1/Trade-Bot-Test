"""
Tradebot V1 — BERTurk Duygu Analizi Modeli
============================================
Hugging Face transformers ile BERTurk (dbmdz/bert-base-turkish-cased)
fine-tune edilmiş Türkçe finansal haber duygu analizi modeli.

3 sınıf: negatif, notr, pozitif

Kullanım:
  from news.sentiment.berturk_model import BERTurkSentiment
  model = BERTurkSentiment()
  result = model.predict("şirket rekor kar açıkladı")
"""

import os
import json
import numpy as np

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "berturk_sentiment",
)
LABELS = ["negatif", "notr", "pozitif"]
LABEL_TO_SCORE = {
    "negatif": 20,
    "notr": 50,
    "pozitif": 80,
}


def _detect_num_labels() -> int:
    """Kaydedilmiş modelin kaç sınıf kullandığını tespit et."""
    meta_path = os.path.join(MODEL_DIR, "training_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("num_labels", 3)
        except Exception:
            pass
    # config.json'dan dene
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return cfg.get("num_labels", 3)
        except Exception:
            pass
    return 3


class BERTurkSentiment:
    """BERTurk fine-tune duygu analizi modeli (3-class)."""

    BASE_MODEL = "dbmdz/bert-base-turkish-cased"
    MAX_LENGTH = 128

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._loaded = False
        self._num_labels = None

    @property
    def available(self) -> bool:
        """Model dosyaları mevcut mu?"""
        return os.path.exists(MODEL_DIR) and (
            os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))
            or os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin"))
        )

    def _load(self) -> bool:
        """Model ve tokenizer'ı yükle (lazy loading)."""
        if self._loaded:
            return True

        if not self.available:
            print("[BERTurk] Model dosyaları bulunamadı. Eğitim gerekli.")
            return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._num_labels = _detect_num_labels()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            print(
                f"[BERTurk] Model yüklendi ({self._device}, {self._num_labels}-class)"
            )
            return True
        except ImportError:
            print(
                "[BERTurk] transformers kütüphanesi yüklü değil: pip install transformers"
            )
            return False
        except Exception as e:
            print(f"[BERTurk] Model yükleme hatası: {e}")
            return False

    def _probs_to_result(self, probs) -> dict:
        """Softmax olasılıklarını sonuca dönüştür."""
        n = len(probs)

        if n == 3:
            # 3-class: negatif(0), notr(1), pozitif(2)
            labels = LABELS
            score_map = LABEL_TO_SCORE
        elif n == 5:
            # Eski 5-class model — 3-class'a map et
            old_labels = ["cok_negatif", "negatif", "notr", "pozitif", "cok_pozitif"]
            old_scores = [10, 30, 50, 70, 90]
            weighted_score = sum(probs[i] * old_scores[i] for i in range(5))
            # 3-class'a kümele
            neg_prob = probs[0] + probs[1]
            notr_prob = probs[2]
            pos_prob = probs[3] + probs[4]
            new_probs = [neg_prob, notr_prob, pos_prob]
            idx = int(np.argmax(new_probs))
            label = LABELS[idx]
            confidence = float(new_probs[idx])
            return {
                "score": round(float(weighted_score), 1),
                "label": label,
                "confidence": round(confidence, 3),
                "model": "berturk",
                "probabilities": {
                    LABELS[i]: round(float(new_probs[i]), 3) for i in range(3)
                },
            }
        else:
            labels = LABELS[:n] if n <= 3 else LABELS
            score_map = {l: LABEL_TO_SCORE.get(l, 50) for l in labels}

        idx = int(np.argmax(probs))
        label = labels[idx] if idx < len(labels) else "notr"
        confidence = float(probs[idx])
        weighted_score = sum(
            probs[i] * score_map.get(labels[i], 50) for i in range(n) if i < len(labels)
        )

        return {
            "score": round(float(weighted_score), 1),
            "label": label,
            "confidence": round(confidence, 3),
            "model": "berturk",
            "probabilities": {
                labels[i]: round(float(probs[i]), 3)
                for i in range(n)
                if i < len(labels)
            },
        }

    def predict(self, text: str) -> dict:
        """
        Tek metin için tahmin.

        Returns:
            {"score": float, "label": str, "confidence": float, "model": "berturk"}
        """
        if not self._load():
            return {"score": 50, "label": "notr", "confidence": 0.0, "model": "berturk"}

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            return self._probs_to_result(probs)

        except Exception as e:
            print(f"[BERTurk] Tahmin hatası: {e}")
            return {"score": 50, "label": "notr", "confidence": 0.0, "model": "berturk"}

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Toplu tahmin."""
        if not self._load():
            return [
                {"score": 50, "label": "notr", "confidence": 0.0, "model": "berturk"}
            ] * len(texts)

        try:
            import torch

            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    all_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                for probs in all_probs:
                    results.append(self._probs_to_result(probs))

            return results
        except Exception as e:
            print(f"[BERTurk] Batch tahmin hatası: {e}")
            return [
                {"score": 50, "label": "notr", "confidence": 0.0, "model": "berturk"}
            ] * len(texts)
