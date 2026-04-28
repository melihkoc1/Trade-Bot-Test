import sys

print("Python:", sys.version)

# Test 1: Database modülü
print("\n--- Test 1: Database ---")
from news.database import NewsDatabase

db = NewsDatabase()
stats = db.get_stats()
print(f"DB oluşturuldu: {stats}")

# Test 2: Fetcher modülleri
print("\n--- Test 2: Fetchers ---")
from news.fetchers.base_fetcher import BaseFetcher
from news.fetchers.investing_fetcher import InvestingFetcher
from news.fetchers.kap_fetcher import KAPFetcher
from news.fetchers.bigpara_fetcher import BigparaFetcher
from news.fetchers.foreks_fetcher import ForeksFetcher
from news.fetchers.tcmb_fetcher import TCMBFetcher

print("Tüm fetcher modülleri import edildi")

inv = InvestingFetcher()
print(f"InvestingFetcher: {inv}")
print(f"  RSS Feeds: {list(inv.RSS_FEEDS.keys())}")
print(f"  Synonym sayısı: {len(inv.SYMBOL_SYNONYMS)}")

# Test 3: Classifier
print("\n--- Test 3: Classifier ---")
from news.classifier import NewsClassifier

clf = NewsClassifier()
test_news = {
    "title": "Garanti BBVA rekor kar açıkladı",
    "summary": "Banka sektöründe güçlü bilanço",
}
result = clf.classify(test_news)
print(
    f"Classifier sonucu: type={result['news_type']}, symbols={result.get('symbols', [])}"
)

# Test 4: Macro Impact
print("\n--- Test 4: Macro Impact ---")
from news.macro_impact import MacroImpactCalculator

mac = MacroImpactCalculator()
test_macro = {"category": "faiz"}
impacts = mac.calculate_impact(test_macro, -30)
print(f"Faiz artışı etkileri: {len(impacts)} sektör etkilendi")
for imp in impacts[:3]:
    print(f"  {imp['sector']}: {imp['impact_direction']} ({imp['impact_score']:.3f})")

# Test 5: Sentiment modülleri
print("\n--- Test 5: Sentiment ---")
from news.sentiment.gemini_analyzer import GeminiAnalyzer
from news.sentiment.berturk_model import BERTurkSentiment
from news.sentiment.ensemble import SentimentEnsemble

gemini = GeminiAnalyzer()
berturk = BERTurkSentiment()
ensemble = SentimentEnsemble()
print(f"Gemini available: {gemini.available}")
print(f"BERTurk available: {berturk.available}")
print(f"Engines: {ensemble.engines_status}")

# Test 6: NewsManager
print("\n--- Test 6: NewsManager ---")
from news.news_manager import NewsManager, get_sentiment_score

nm = NewsManager(use_db=True)
print(f"NewsManager oluşturuldu")
print(f"  Fetchers: {list(nm.fetchers.keys())}")
print(f"  DB: {nm.db is not None}")
print(f"  Stats: {nm.get_stats()}")

# Test 7: Scheduler
print("\n--- Test 7: Scheduler ---")
from news.scheduler import NewsScheduler

sched = NewsScheduler(nm)
print(f"Scheduler status: {sched.get_status()}")

print("\n=== TÜM TESTLER BAŞARILI ===")
