import feedparser
import requests
import os
import time
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from config import SECTOR_MAP

load_dotenv()

class RSSNewsScraper:
    def __init__(self):
        self.rss_feeds = {
            "investing_genel": "https://tr.investing.com/rss/news.rss",
            "investing_borsa": "https://tr.investing.com/rss/news_25.rss"
        }
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Sektörel anahtar kelimeler (config.py'deki sektörlerle uyumlu)
        self.sector_keywords = {
            "Banka": ["Banka", "Akbank", "Garanti", "İş Bankası", "Yapı Kredi", "Vakıfbank", "Halkbank", "Faiz", "Kredi"],
            "Enerji": ["Enerji", "Elektrik", "Petrol", "Doğalgaz", "Güneş", "Rüzgar", "Tüpraş", "Yenilenebilir"],
            "Ulasim": ["Ulaşım", "Hava Yolu", "THY", "Pegasus", "TAV", "Lojistik", "Havalimanı"],
            "Otomotiv": ["Otomotiv", "Araç", "Üretim", "Ford", "Tofaş", "Doğuş", "Lastik"],
            "DemirCelik": ["Demir", "Çelik", "Ereğli", "Kardemir", "Metal", "Cevher"],
            "Holding": ["Holding", "Koç", "Sabancı", "Şişecam", "Alarko", "İştirak"],
            "Gıda": ["Gıda", "Perakende", "Market", "BİM", "Migros", "Şok", "Tarım"],
            "Teknoloji": ["Teknoloji", "Yazılım", "Bilişim", "Savunma", "Aselsan", "Çip"]
        }
        
    def fetch_rss_news(self, symbol=None, sector=None):
        """RSS beslemelerinden haberleri çeker ve filtreler (Sembol veya Sektör bazlı)."""
        all_entries = []
        clean_symbol = symbol.replace(".IS", "").upper() if symbol else None
        
        # Sembol eşanlamlıları
        synonyms = {
            "THYAO": ["THY", "TURK HAVA YOLLARI", "TÜRK HAVA YOLLARI"],
            "ASELS": ["ASELSAN"],
            "EREGL": ["EREĞLİ", "EREGLI", "ERDEMİR"],
            "KCHOL": ["KOÇ HOLDİNG", "KOC HOLDING"],
            "SAHOL": ["SABANCI HOLDİNG", "SABANCI HOLDING"],
            "SISE": ["ŞİŞECAM", "SISECAM"],
            "AKBNK": ["AKBANK"],
            "GARAN": ["GARANTİ", "GARANTI BBVA"],
            "TUPRS": ["TÜPRAŞ", "TUPRAS"]
        }
        
        search_terms = []
        if clean_symbol:
            search_terms = [clean_symbol]
            if clean_symbol in synonyms:
                search_terms.extend(synonyms[clean_symbol])
        elif sector:
            # Sektör bazlı anahtar kelimeleri getir
            search_terms = self.sector_keywords.get(sector, [sector])

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

        for name, url in self.rss_feeds.items():
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    continue
                
                feed = feedparser.parse(response.content)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    description = entry.get("description", "")
                    full_text = f"{title} {summary} {description}"
                    
                    # KAP tespiti
                    is_kap = False
                    if "KAP:" in title.upper() or "KAP " in title.upper() or "KAMUYU AYDINLATMA" in full_text.upper():
                        is_kap = True

                    item = {
                        "source": name,
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": summary if summary else description[:200],
                        "sentiment": "Neutral",
                        "score": 50,
                        "is_kap": is_kap,
                        "type": "Haber"
                    }
                    
                    found = False
                    if not search_terms:
                        found = True
                    else:
                        for term in search_terms:
                            if term.upper() in full_text.upper():
                                found = True
                                break
                    
                    if found:
                        item["type"] = "KAP" if is_kap else ("Sektör" if sector else "Haber")
                        all_entries.append(item)
            except Exception as e:
                print(f"RSS Fetch Error ({name}): {e}")
        
        unique_news = {item["link"]: item for item in all_entries}.values()
        try:
            sorted_news = sorted(unique_news, key=lambda x: x.get("published", ""), reverse=True)
        except:
            sorted_news = list(unique_news)
            
        return sorted_news[:10] 

    def _analyze_sentiment_lstm(self, title: str) -> tuple:
        """
        Bidirectional LSTM ile Türkçe haber başlığını sınıflandırır.
        Returns: (score 0-100, sentiment str)
        """
        try:
            from sentiment_lstm import get_sentiment
            label, conf = get_sentiment(title)
            label_to_score = {"POZITIF": 75, "NOTR": 50, "NEGATIF": 25}
            base_score = label_to_score.get(label, 50)
            # Guven oranina gore 50'den uzaklastir
            score = 50 + (base_score - 50) * conf
            sentiment_map = {"POZITIF": "Positive", "NOTR": "Neutral", "NEGATIF": "Negative"}
            return round(score), sentiment_map.get(label, "Neutral")
        except Exception:
            return 50, "Neutral"

    def analyze_sentiment_gemini(self, title, summary, symbol, news_type="Haber"):
        """OpenRouter üzerinden Gemini ile duygu analizi yapar. Başarısız olursa LSTM kullanır."""
        if not self.api_key:
            return self._analyze_sentiment_lstm(title)
        
        context_prompt = ""
        if news_type == "KAP":
            context_prompt = "Bu bir resmi KAP (Kamuyu Aydınlatma Platformu) bildirimidir, etkisi daha yüksektir."
        elif news_type == "Sektör":
            context_prompt = f"Bu haber doğrudan şirket hakkında olmayabilir ancak şirketin faaliyet gösterdiği sektörü etkileyebilir."

        prompt = f"""Sen kıdemli bir borsa analistisin. Aşağıdaki gelişmeyi oku.
{context_prompt}

İlgili Şirket: {symbol}
Başlık: {title}
Özet: {summary[:500]}

GÖREV: Bu haberin/bildirimin {symbol} hisse fiyatına KISA-ORTA vadede etkisini puanla.
- 0: Çok Negatif (İflas, çok kötü zarar, büyük ceza vb.)
- 50: Nötr (Etkisi belirsiz veya rutin haber)
- 100: Çok Pozitif (Büyük ihale alımı, bedelsiz sermaye artırımı, çok iyi bilanço vb.)

SADECE 0-100 arasında bir sayı döndür. Açıklama yapma."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.1,
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                match = re.search(r'\d+', content)
                if match:
                    score = int(match.group())
                    score = max(0, min(100, score))
                    
                    sentiment = "Neutral"
                    if score > 60: sentiment = "Positive"
                    elif score < 40: sentiment = "Negative"
                    return score, sentiment
        except Exception as e:
            print(f"Sentiment Analysis Error: {e}")

        # Gemini başarısız → LSTM fallback
        return self._analyze_sentiment_lstm(title)

# Singleton instance
scraper = RSSNewsScraper()

def get_sentiment_score(symbol):
    """strategy.py ve app.py ile uyumluluk için ana fonksiyon."""
    # 1. Hisseye özel haberleri ara
    news_items = scraper.fetch_rss_news(symbol=symbol)
    
    # 2. Eğer yoksa, SEKTÖRE özel haberleri ara
    if not news_items:
        sector = SECTOR_MAP.get(symbol)
        if sector:
            news_items = scraper.fetch_rss_news(sector=sector)
            
    # 3. Eğer o da yoksa, Genel haberleri getir (Son çare)
    if not news_items:
        news_items = scraper.fetch_rss_news(symbol=None)
        if not news_items:
            return 0, []

    total_score = 0
    count = 0
    final_news = []
    
    for item in news_items[:3]:
        score, sentiment = scraper.analyze_sentiment_gemini(
            item["title"], 
            item["summary"], 
            symbol, 
            news_type=item.get("type", "Haber")
        )
        item["score"] = score
        item["sentiment"] = sentiment
        total_score += score
        count += 1
        final_news.append(item)
        time.sleep(0.5) 
        
    avg_score = total_score / count if count > 0 else 50
    normalized_score = avg_score - 50
    
    # Sektör haberi ise strateji etkisini azaltalım (%50 ağırlık)
    if any(n.get("type") == "Sektör" for n in final_news):
        normalized_score *= 0.5
        
    return normalized_score, final_news

if __name__ == "__main__":
    test_symbol = "THYAO.IS"
    print(f"--- {test_symbol} Haber ve KAP Analizi Testi ---")
    score, news = get_sentiment_score(test_symbol)
    print(f"Normalleştirilmiş Skor: {score}")
    for n in news:
        type_label = n.get('type', 'Haber')
        print(f"[{type_label} | {n['sentiment']}] {n['title']} (Skor: {n['score']})")
