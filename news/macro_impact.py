"""
Tradebot V1 — Makro Haber Etki Hesaplayıcı
============================================
Makro haberler (faiz, enflasyon, döviz, petrol, altın) için
sektör bazlı etki hesaplama. config.py'deki SECTOR_MACRO_CORRELATION
tablosuyla uyumlu çalışır.
"""

import re
from config import SECTOR_MACRO_CORRELATION, SECTOR_MAP


class MacroImpactCalculator:
    """Makro haberlerin sektör/hisse bazlı etkisini hesaplar."""

    # Makro haber kategorisi → config.py macro key eşleşmesi
    CATEGORY_TO_MACRO = {
        "faiz": None,  # Faiz özel hesaplanır (yükseliş/düşüş farklı)
        "enflasyon": None,  # Enflasyon özel
        "doviz": "dolar",
        "petrol": "petrol",
        "altin": "altin",
        "buyume": "bist100",  # Büyüme genel endeksi etkiler
        "istihdam": "bist100",
        "kur": "dolar",
        "likidite": None,
    }

    # Faiz kararı etki matrisi (yön bağımlı)
    FAIZ_IMPACT = {
        "artis": {  # Faiz artışı
            "Banka": -0.5,  # Kısa vadede olumsuz, uzun vadede olumlu olabilir
            "GYO": -0.7,
            "Insaat": -0.6,
            "Holding": -0.3,
            "Perakende": -0.4,
            "Enerji": -0.2,
        },
        "indirim": {  # Faiz indirimi
            "Banka": 0.5,
            "GYO": 0.7,
            "Insaat": 0.6,
            "Holding": 0.4,
            "Perakende": 0.3,
            "Teknoloji": 0.3,
        },
    }

    # Enflasyon etki matrisi
    ENFLASYON_IMPACT = {
        "yuksek": {  # Yüksek enflasyon
            "Perakende": -0.5,
            "Banka": -0.3,
            "Enerji": 0.2,  # Enerji fiyatları artar
            "Madencilik": 0.3,  # Altın hedge
        },
        "dusuk": {  # Düşük enflasyon (iyi haber)
            "Perakende": 0.4,
            "Banka": 0.3,
            "Holding": 0.2,
        },
    }

    def calculate_impact(self, news_item: dict, sentiment_score: float) -> list[dict]:
        """
        Makro haberin sektör bazlı etkisini hesapla.

        Args:
            news_item: Haber dict'i (category alanı gerekli)
            sentiment_score: Haberin duygu skoru (-100 ile +100 arası)

        Returns:
            [{"sector": "Banka", "impact_direction": "negative",
              "impact_score": -0.6, "macro_type": "faiz"}, ...]
        """
        category = news_item.get("category", "")
        if not category:
            return []

        impacts = []

        # Faiz kararı — yön bağımlı
        if category == "faiz":
            impacts = self._calculate_faiz_impact(sentiment_score)

        # Enflasyon — yön bağımlı
        elif category == "enflasyon":
            impacts = self._calculate_enflasyon_impact(sentiment_score)

        # Diğer makro haberler — SECTOR_MACRO_CORRELATION kullan
        else:
            macro_key = self.CATEGORY_TO_MACRO.get(category)
            if macro_key:
                impacts = self._calculate_from_correlation(
                    macro_key, sentiment_score, category
                )

        return impacts

    def get_impact_for_symbol(self, symbol: str, impacts: list[dict]) -> float:
        """
        Bir sembol için toplam makro etkiyi hesapla.

        Args:
            symbol: Hisse sembolü (ör: "THYAO.IS")
            impacts: calculate_impact() çıktısı

        Returns:
            Toplam etki skoru (-1 ile +1 arası)
        """
        sector = SECTOR_MAP.get(symbol)
        if not sector:
            return 0.0

        total = 0.0
        count = 0
        for impact in impacts:
            if impact["sector"] == sector:
                total += impact["impact_score"]
                count += 1

        return total / count if count > 0 else 0.0

    def get_all_sector_impacts(self, impacts: list[dict]) -> dict:
        """
        Tüm sektörler için etki özetini döndür.

        Returns:
            {"Banka": -0.5, "Enerji": 0.3, ...}
        """
        sector_scores = {}
        for impact in impacts:
            sector = impact["sector"]
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append(impact["impact_score"])

        return {
            sector: sum(scores) / len(scores)
            for sector, scores in sector_scores.items()
        }

    def _calculate_faiz_impact(self, sentiment_score: float) -> list[dict]:
        """Faiz kararı etkisi. sentiment_score > 0 ise artış haberi."""
        if sentiment_score > 10:
            # Faiz artışı veya beklentinin üzeri
            matrix = self.FAIZ_IMPACT["artis"]
        elif sentiment_score < -10:
            # Faiz indirimi veya beklentinin altı
            matrix = self.FAIZ_IMPACT["indirim"]
        else:
            return []  # Beklenti dahilinde, etki az

        return [
            {
                "sector": sector,
                "impact_direction": "positive" if score > 0 else "negative",
                "impact_score": score * (abs(sentiment_score) / 100),
                "macro_type": "faiz",
            }
            for sector, score in matrix.items()
        ]

    def _calculate_enflasyon_impact(self, sentiment_score: float) -> list[dict]:
        """Enflasyon verisi etkisi."""
        if sentiment_score < -10:
            # Yüksek enflasyon (olumsuz haber)
            matrix = self.ENFLASYON_IMPACT["yuksek"]
        elif sentiment_score > 10:
            # Düşük enflasyon (olumlu haber)
            matrix = self.ENFLASYON_IMPACT["dusuk"]
        else:
            return []

        return [
            {
                "sector": sector,
                "impact_direction": "positive" if score > 0 else "negative",
                "impact_score": score * (abs(sentiment_score) / 100),
                "macro_type": "enflasyon",
            }
            for sector, score in matrix.items()
        ]

    def _calculate_from_correlation(
        self, macro_key: str, sentiment_score: float, category: str
    ) -> list[dict]:
        """SECTOR_MACRO_CORRELATION tablosundan etki hesapla."""
        impacts = []
        direction = 1 if sentiment_score > 0 else -1

        for sector, correlations in SECTOR_MACRO_CORRELATION.items():
            if macro_key in correlations:
                corr = correlations[macro_key]
                # Korelasyon * yön * normalleştirilmiş skor
                raw_impact = corr * direction * (abs(sentiment_score) / 100)

                impacts.append(
                    {
                        "sector": sector,
                        "impact_direction": "positive"
                        if raw_impact > 0
                        else "negative",
                        "impact_score": round(raw_impact, 3),
                        "macro_type": category,
                    }
                )

        return impacts

    def detect_macro_category(self, text: str) -> tuple:
        """
        Haber metninden makro kategori ve yön tespit et.

        Returns:
            (category, direction)
            category: "faiz" | "enflasyon" | "doviz" | "petrol" | "altin" | "buyume" | None
            direction: "artis" | "indirim" | None
        """
        text_lower = text.lower()

        ARTIS_WORDS = [
            "artırdı", "arttı", "artış", "yükseldi", "yükseltildi",
            "çıktı", "çıkardı", "sıkılaştırdı", "şahin", "zam", "yukarı",
        ]
        INDIRIM_WORDS = [
            "indirdi", "indi", "indirim", "düşürdü", "düştü", "azalttı",
            "gevşetti", "güvercin", "kesim", "kesinti", "aşağı",
        ]

        def yon_bul(ekstra_artis=None, ekstra_indirim=None):
            artis = ARTIS_WORDS + (ekstra_artis or [])
            indirim = INDIRIM_WORDS + (ekstra_indirim or [])
            for w in artis:
                if w in text_lower:
                    return "artis"
            for w in indirim:
                if w in text_lower:
                    return "indirim"
            return None

        # Faiz
        for w in ["faiz", "politika faizi", "merkez bankası", "tcmb", "baz puan"]:
            if w in text_lower:
                return "faiz", yon_bul()

        # Enflasyon
        for w in ["enflasyon", "tüfe", "üfe", "fiyat endeksi"]:
            if w in text_lower:
                return "enflasyon", yon_bul(
                    ekstra_artis=["yüksek", "rekor", "beklentinin üzerinde"],
                    ekstra_indirim=["geriledi", "azaldı", "beklentinin altında"],
                )

        # Döviz
        for w in ["dolar", "euro", "döviz", "kur", "liranın"]:
            if w in text_lower:
                return "doviz", yon_bul(
                    ekstra_artis=["güçlendi", "zirve"],
                    ekstra_indirim=["değer kaybetti", "zayıfladı"],
                )

        # Petrol
        for w in ["petrol", "brent", "ham petrol", "opec"]:
            if w in text_lower:
                return "petrol", yon_bul()

        # Altın
        for w in ["altın", "ons altın", "gram altın"]:
            if w in text_lower:
                return "altin", yon_bul()

        # Büyüme
        for w in ["büyüme", "gdp", "gsyih", "resesyon", "daralma"]:
            if w in text_lower:
                return "buyume", yon_bul(
                    ekstra_artis=["büyüdü", "güçlü", "pozitif"],
                    ekstra_indirim=["daraldı", "negatif", "resesyon"],
                )

        return None, None

    def analyze_news_text(self, text: str, sentiment_score: float = 0.0) -> dict:
        """
        Ham haber metninden makro kategori + yön tespiti yapıp
        sektör bazlı etki hesapla.

        Args:
            text: Haber başlığı veya metni
            sentiment_score: NLP modelinden gelen skor (-100..+100)

        Returns:
            {
                "category": "faiz",
                "direction": "indirim",
                "sector_impacts": {"Banka": 0.35, "GYO": 0.49, ...},
                "summary": "Faiz (indirim): GYO (+0.42), Banka (+0.30), ..."
            }
        """
        category, direction = self.detect_macro_category(text)

        if not category:
            return {
                "category": None,
                "direction": None,
                "sector_impacts": {},
                "summary": "Makro kategori tespit edilemedi.",
            }

        # sentiment_score verilmediyse yönden tahmin et
        if sentiment_score == 0.0:
            if direction == "artis":
                sentiment_score = 60.0
            elif direction == "indirim":
                sentiment_score = -60.0

        impacts = self.calculate_impact({"category": category}, sentiment_score)
        sector_impacts = self.get_all_sector_impacts(impacts)

        # Özet metin
        if sector_impacts:
            parts = []
            for sector, score in sorted(sector_impacts.items(), key=lambda x: -abs(x[1])):
                sign = "+" if score > 0 else ""
                parts.append(f"{sector} ({sign}{score:.2f})")
            summary = f"{category.capitalize()} ({direction}): " + ", ".join(parts)
        else:
            summary = f"{category.capitalize()} haberi tespit edildi, etki hesaplanamadı."

        return {
            "category": category,
            "direction": direction,
            "sector_impacts": sector_impacts,
            "summary": summary,
        }
