"""
Tradebot V1 — BERTurk Fine-tune Eğitim Script'i
=================================================
Türkçe finansal haber başlıklarını sınıflandırmak için
BERTurk (dbmdz/bert-base-turkish-cased) modelini fine-tune eder.

3 sınıf: negatif (0), notr (1), pozitif (2)

Kullanım:
  python -m news.sentiment.train_berturk --train
  python -m news.sentiment.train_berturk --train --epochs 5 --lr 2e-5
  python -m news.sentiment.train_berturk --evaluate
  python -m news.sentiment.train_berturk --collect --count 2000
  python -m news.sentiment.train_berturk --collect --hf --count 50000
  python -m news.sentiment.train_berturk --label
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from datetime import datetime

# Proje kök dizinini path'e ekle
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "berturk_sentiment")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "news_training")
LABELS = ["negatif", "notr", "pozitif"]
NUM_LABELS = 3

# ─────────────────────────────────────────────────────────
# Başlangıç eğitim verisi (3-class seed data)
# ─────────────────────────────────────────────────────────
SEED_DATA = [
    # POZITIF (2)
    ("Şirket net karını üçe katladı, beklentilerin çok üzerinde geldi", 2),
    ("Bedelsiz sermaye artırımı kararı açıklandı hisse uçtu", 2),
    ("Dev ihale kazanıldı şirket değeri rekor kırdı", 2),
    ("Yabancı dev şirket ile stratejik ortaklık anlaşması imzalandı", 2),
    ("Rekor temettü dağıtım kararı yatırımcıları sevindirdi", 2),
    ("Uluslararası kredi notu iki kademe yükseltildi", 2),
    ("Şirket satış gelirlerini yüzde yüz artırdı", 2),
    ("Borsa tüm zamanların en yüksek kapanışını yaptı", 2),
    ("Şirkete milyar dolarlık yatırım taahhüdü alındı", 2),
    ("Hisse senedi bölünme kararı açıklandı yatırımcılar memnun", 2),
    ("Net kar marjı sektör ortalamasının üç katına çıktı", 2),
    ("Şirket karını yüzde otuz artırdı", 2),
    ("Güçlü büyüme rakamları açıklandı", 2),
    ("İhracat geçen yıla göre arttı", 2),
    ("Analistler hedef fiyatı yükseltti", 2),
    ("Şirket yeni fabrika yatırımı yapıyor", 2),
    ("Olumlu bilanço açıklandı", 2),
    ("Borsa güne yükselişle başladı", 2),
    ("Yabancı yatırımcı alımları artıyor", 2),
    ("Faiz kararı piyasaları rahatlattı", 2),
    ("Büyüme tahminleri yukarı revize edildi", 2),
    ("Temettü ödemesi artırıldı", 2),
    ("Enflasyon beklentinin altında geldi piyasalar pozitif", 2),
    ("Hisse güçlü al tavsiyesi aldı", 2),
    ("Sektörde talep artışı yaşandı", 2),
    ("Yeni yatırım teşviki alındı şirket faydalanacak", 2),
    ("Analistler hedef fiyatı yüzde elli yukarı revize etti", 2),
    ("Şirket yeni pazarlara açılarak cirosunu ikiye katladı", 2),
    ("Mega proje ihalesi kazanıldı büyüme beklentileri arttı", 2),
    ("THY yolcu sayısında rekor kırdı gelirler arttı", 2),
    ("ASELSAN savunma sanayi sözleşmesi imzaladı", 2),
    ("Şirket yurt dışına açılıyor ihracat hedefi büyüdü", 2),
    ("Kar payı ödemesi beklentinin üzerinde açıklandı", 2),
    ("Şirketin pazar payı artıyor rakiplerine üstünlük sağlıyor", 2),
    ("İştirak satışından büyük kazanç elde edildi", 2),
    ("Kredi derecelendirme kuruluşu notu artırdı", 2),
    ("Şirket borçsuz bilanço ile büyümeye devam ediyor", 2),
    ("Yeni ürün lansmanı büyük ilgi gördü satışlar patladı", 2),
    ("Sözleşme yenilendi uzun vadeli gelir güvence altında", 2),
    ("Şirket operasyonel verimliliğini artırdı", 2),
    # NOTR (1)
    ("Şirket yönetim kurulu toplandı rutin gündem", 1),
    ("Faiz kararı beklentiler dahilinde geldi", 1),
    ("Borsa gününü yatay kapattı", 1),
    ("Beklentiler değişmedi piyasalar sakin", 1),
    ("Şirket genel kurul tarihi belirlendi", 1),
    ("Piyasalar temkinli seyrediyor", 1),
    ("Analistler bekle tavsiyesini korudu", 1),
    ("Sektör raporu yayımlandı önemli değişiklik yok", 1),
    ("Şirket yıllık raporunu yayımladı", 1),
    ("Piyasalar dolar kuru takibinde sakin", 1),
    ("Borsa günlük işlem hacmi ortalama seyretti", 1),
    ("Şirketin çeyrek sonuçları bekleniyor", 1),
    ("Piyasalar ABD verilerini bekliyor", 1),
    ("Şirket bağımsız denetim sürecini tamamladı", 1),
    ("Merkez bankası toplantısı yaklaşıyor piyasalar beklemede", 1),
    ("Şirket yeni CEO atadı geçiş süreci başladı", 1),
    ("Borsa haftayı küçük değişimle kapattı", 1),
    ("Şirket faaliyet raporunu açıkladı normal seyir", 1),
    ("Piyasalar seçim öncesi bekleme modunda", 1),
    ("Şirket stratejik planını güncelledi", 1),
    ("TCMB faiz kararını açıkladı beklenti dahilinde", 1),
    ("Şirket olağan genel kurulunu tamamladı", 1),
    ("Borsa İstanbul işlem saatlerini duyurdu", 1),
    ("SPK haftalık bültenini yayımladı", 1),
    ("Şirket bağlı ortaklığında hisse devretti", 1),
    ("Piyasalarda likidite normal seyrediyor", 1),
    ("Şirket yönetim kurulunda görev değişikliği", 1),
    ("Enflasyon verisi açıklandı beklentilerle uyumlu", 1),
    ("Dolar kuru güne yatay başladı", 1),
    ("Şirket sermaye artırımı için SPK onayı bekleniyor", 1),
    # NEGATIF (0)
    ("Şirket beklentilerin altında kar açıkladı", 0),
    ("Kur dalgalanması şirket maliyetlerini artırdı", 0),
    ("Faiz artışı hisseleri baskıladı", 0),
    ("Hisse sert düştü yüzde beş değer kaybetti", 0),
    ("Şirkete vergi cezası kesildi", 0),
    ("Üretimde yavaşlama yaşandı", 0),
    ("İhracatta gerileme kaydedildi", 0),
    ("Borçlanma maliyetleri arttı", 0),
    ("Piyasalar satış baskısı altında", 0),
    ("Büyüme tahminleri aşağı revize edildi", 0),
    ("Şirket yöneticisi istifa etti belirsizlik arttı", 0),
    ("Talep düşüşü yaşandı sektör baskı altında", 0),
    ("Döviz kurlarındaki artış şirket bilançosunu olumsuz etkiledi", 0),
    ("Analistler hedef fiyatı düşürdü", 0),
    ("Çalışan sayısı azaltılıyor tasarruf önlemleri alınıyor", 0),
    ("Sektörde rekabet artıyor kar marjları baskı altında", 0),
    ("Hisse devre kesici çalıştırdı düşüş sert oldu", 0),
    ("Enflasyon beklentilerin üzerinde geldi piyasalar olumsuz", 0),
    ("Kredi notu görünümü negatife çevrildi", 0),
    ("Şirketin pazar payı azalıyor", 0),
    ("Şirket konkordato ilan etti iflas kapıda", 0),
    ("Şirkete milyarlık vergi cezası ve soruşturma", 0),
    ("Yönetim kurulu başkanı gözaltına alındı dolandırıcılık", 0),
    ("Şirket iflas başvurusu yaptı borçlar ödenemiyor", 0),
    ("Hisseler çöktü yüzde yirmi değer kaybetti", 0),
    ("Şirketin tüm varlıklarına haciz konuldu", 0),
    ("SPK şirkete işlem yasağı getirdi manipülasyon", 0),
    ("Krediler dondu şirket nakit krizi yaşıyor", 0),
    ("Şirket operasyonlarını tamamen durdurdu kapanıyor", 0),
    ("Büyük yolsuzluk skandalı ortaya çıktı yöneticiler tutuklandı", 0),
    ("Şirket piyasa değerinin yarısını kaybetti", 0),
    ("Borsa panikle sert düştü devreler çalışıyor", 0),
    ("Sektörde büyük kriz baş gösterdi şirketler zor durumda", 0),
    ("Şirket varlıklarını satarak borç kapatmaya çalışıyor", 0),
    ("Net zarar beklentinin çok üzerinde geldi yatırımcılar şokta", 0),
    ("Fabrika yangınında üretim durdu ciddi hasar var", 0),
    ("Şirkete SPK soruşturması açıldı manipülasyon iddiaları", 0),
    ("Döviz krizi bilançoyu vurdu şirket zarar açıkladı", 0),
    ("Banka takipteki kredilerini artırdı sorunlu portföy büyüdü", 0),
    ("Şirket tesislerinde iş kazası üretim durdu", 0),
    ("Yüksek hammadde maliyeti kar marjını sıfırladı", 0),
]

# HuggingFace dataset etiket eşlemesi (256k dataset → 3 sınıf)
HF_LABEL_MAP = {
    "Negatif": 0,
    "Nötr": 1,
    "Pozitif": 2,
    # küçük harf varyantları
    "negatif": 0,
    "nötr": 1,
    "pozitif": 2,
    "notr": 1,
}


def load_huggingface_dataset(max_samples: int = 50000) -> tuple[list, list]:
    """
    HuggingFace'den ituperceptron/turkish-financial-sentiment-256k indir.
    max_samples kadar örnek döndürür (dengeli örnekleme).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[HF] 'datasets' kütüphanesi yüklü değil: pip install datasets")
        return [], []

    print(f"[HF] Dataset indiriliyor: ituperceptron/turkish-financial-sentiment-256k")
    print(f"[HF] Hedef: {max_samples} örnek")

    try:
        ds = load_dataset(
            "ituperceptron/turkish-financial-sentiment-256k",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[HF] Dataset indirme hatası: {e}")
        return [], []

    texts = []
    labels = []

    # Sınıf başına dengeli örnekleme
    per_class = max_samples // NUM_LABELS
    class_counts = {i: 0 for i in range(NUM_LABELS)}

    # Dataset'i karıştır
    import random

    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)

    for idx in indices:
        row = ds[idx]
        label_str = row.get("label", "").strip()
        label_int = HF_LABEL_MAP.get(label_str)
        if label_int is None:
            continue

        text = row.get("text", "").strip()
        if not text or len(text) < 10:
            continue

        # Maksimum uzunluk kontrolü (çok uzun metinleri kırp)
        if len(text) > 512:
            text = text[:512]

        if class_counts[label_int] < per_class:
            texts.append(text)
            labels.append(label_int)
            class_counts[label_int] += 1

        if sum(class_counts.values()) >= max_samples:
            break

    print(f"[HF] Yüklendi: {len(texts)} örnek")
    for i, name in enumerate(LABELS):
        print(f"  {name}: {class_counts[i]}")

    return texts, labels


def collect_headlines(count: int = 2000, use_hf: bool = False) -> int:
    """
    RSS kaynaklarından haber başlıklarını topla ve CSV'ye kaydet.
    use_hf=True ise HuggingFace'den de veri çeker.
    """
    raw_path = os.path.join(DATA_DIR, "raw_headlines.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Mevcut başlıkları yükle
    existing = set()
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # header atla
            for row in reader:
                if row:
                    existing.add(row[0])

    print(f"[Collector] Mevcut başlık sayısı: {len(existing)}")
    print(f"[Collector] Hedef: {count} başlık")

    new_headlines = []

    if use_hf:
        # HuggingFace dataset'inden çek
        hf_texts, hf_labels = load_huggingface_dataset(max_samples=count)
        label_names = LABELS
        for text, label_int in zip(hf_texts, hf_labels):
            if text not in existing:
                new_headlines.append(
                    {
                        "title": text,
                        "source": "huggingface",
                        "news_type": "genel",
                        "date": "",
                        "label": label_names[label_int],
                    }
                )
                existing.add(text)
    else:
        # RSS fetcher'lardan haber topla (tüm fetcher'lar)
        from news.fetchers.investing_fetcher import InvestingFetcher
        from news.fetchers.foreks_fetcher import ForeksFetcher
        from news.fetchers.bigpara_fetcher import BigparaFetcher
        from news.fetchers.tcmb_fetcher import TCMBFetcher
        from news.fetchers.kap_fetcher import KAPFetcher

        for FetcherClass in [
            InvestingFetcher,
            ForeksFetcher,
            BigparaFetcher,
            TCMBFetcher,
            KAPFetcher,
        ]:
            try:
                fetcher = FetcherClass()
                news = fetcher.fetch_all()
                for item in news:
                    title = item.get("title", "").strip()
                    if title and title not in existing and len(title) > 15:
                        new_headlines.append(
                            {
                                "title": title,
                                "source": item.get("source", "unknown"),
                                "news_type": item.get("news_type", "genel"),
                                "date": item.get("published_at", ""),
                                "label": "",
                            }
                        )
                        existing.add(title)
            except Exception as e:
                print(f"[Collector] {FetcherClass.__name__} hatası: {e}")

    # CSV'ye yaz
    mode = "a" if os.path.exists(raw_path) else "w"
    with open(raw_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["title", "source", "news_type", "date", "label"])
        for h in new_headlines:
            writer.writerow(
                [h["title"], h["source"], h["news_type"], h["date"], h.get("label", "")]
            )

    print(
        f"[Collector] {len(new_headlines)} yeni başlık eklendi. Toplam: {len(existing)}"
    )
    return len(new_headlines)


def auto_label_with_gemini(batch_size: int = 50) -> int:
    """
    Etiketlenmemiş başlıkları Gemini ile otomatik etiketle (3-class).
    """
    raw_path = os.path.join(DATA_DIR, "raw_headlines.csv")
    if not os.path.exists(raw_path):
        print("[AutoLabel] raw_headlines.csv bulunamadı. Önce --collect çalıştırın.")
        return 0

    from news.sentiment.gemini_analyzer import GeminiAnalyzer

    gemini = GeminiAnalyzer()
    if not gemini.available:
        print("[AutoLabel] Gemini API key ayarlı değil.")
        return 0

    # CSV oku
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Etiketlenmemiş olanları bul
    unlabeled = [r for r in rows if not r.get("label")]
    if not unlabeled:
        print("[AutoLabel] Etiketlenecek başlık kalmadı.")
        return 0

    print(
        f"[AutoLabel] {len(unlabeled)} başlık etiketlenecek (batch_size={batch_size})"
    )
    labeled_count = 0

    # 3-class map (Gemini 5-class döndürebilir, 3'e indirgiyoruz)
    label_3class_map = {
        "cok_negatif": "negatif",
        "negatif": "negatif",
        "notr": "notr",
        "pozitif": "pozitif",
        "cok_pozitif": "pozitif",
    }

    for item in unlabeled[:batch_size]:
        result = gemini.analyze(
            title=item["title"], news_type=item.get("news_type", "genel")
        )
        raw_label = result["label"]
        label_3 = label_3class_map.get(raw_label, "notr")
        score = result["score"]
        confidence = result["confidence"]

        item["label"] = label_3
        item["auto_score"] = str(score)
        item["auto_confidence"] = str(confidence)
        item["needs_review"] = "1" if confidence < 0.6 else "0"
        labeled_count += 1

        if labeled_count % 10 == 0:
            print(
                f"  [{labeled_count}/{min(len(unlabeled), batch_size)}] etiketlendi..."
            )

    # Geri yaz
    with open(raw_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "title",
            "source",
            "news_type",
            "date",
            "label",
            "auto_score",
            "auto_confidence",
            "needs_review",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[AutoLabel] {labeled_count} başlık etiketlendi.")
    needs_review = sum(1 for r in rows if r.get("needs_review") == "1")
    print(f"[AutoLabel] Manuel kontrol gerektiren: {needs_review}")
    return labeled_count


def prepare_dataset(use_hf: bool = False, hf_samples: int = 50000) -> tuple:
    """
    Eğitim veri setini hazırla.
    SEED_DATA + etiketlenmiş CSV verisi + isteğe bağlı HF dataset birleştirilir.
    """
    texts = []
    labels = []

    # 1. Seed data
    for text, label in SEED_DATA:
        texts.append(text)
        labels.append(label)

    # 2. CSV'den etiketli veri (3-class)
    raw_path = os.path.join(DATA_DIR, "raw_headlines.csv")
    if os.path.exists(raw_path):
        label_map = {l: i for i, l in enumerate(LABELS)}
        # eski 5-class etiketleri 3-class'a dönüştür
        five_to_three = {
            "cok_negatif": 0,
            "negatif": 0,
            "notr": 1,
            "pozitif": 2,
            "cok_pozitif": 2,
        }
        with open(raw_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_str = row.get("label", "").strip().lower()
                if label_str in label_map:
                    texts.append(row["title"])
                    labels.append(label_map[label_str])
                elif label_str in five_to_three:
                    texts.append(row["title"])
                    labels.append(five_to_three[label_str])

    # 3. HuggingFace dataset (isteğe bağlı)
    if use_hf:
        hf_texts, hf_labels = load_huggingface_dataset(max_samples=hf_samples)
        texts.extend(hf_texts)
        labels.extend(hf_labels)

    print(f"[Dataset] Toplam: {len(texts)} örnek")
    for i, name in enumerate(LABELS):
        cnt = labels.count(i)
        print(f"  {name}: {cnt} ({cnt / len(texts) * 100:.1f}%)")

    return texts, labels


def train(
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    plot: bool = False,
    use_hf: bool = False,
    hf_samples: int = 50000,
):
    """BERTurk modelini fine-tune et."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as e:
        print(f"[Train] Gerekli kütüphane eksik: {e}")
        print("  pip install transformers torch scikit-learn datasets")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Cihaz: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Veri hazırla
    texts, labels = prepare_dataset(use_hf=use_hf, hf_samples=hf_samples)
    if len(texts) < 50:
        print("[Train] Yetersiz veri! En az 50 örnek gerekli.")
        return

    # Train/Val/Test split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(
        f"[Train] Split: {len(train_texts)} train / {len(val_texts)} val / {len(test_texts)} test"
    )

    # Tokenizer
    # savasy/bert-base-turkish-sentiment-cased: zaten Turkce sentiment icin fine-tune edilmis
    # Uzerine finansal domain fine-tune yapiyoruz -> daha yuksek accuracy
    base_model = "savasy/bert-base-turkish-sentiment-cased"
    print(f"[Train] Model yükleniyor: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    ).to(device)

    # Tokenization önceden yap (dataset'e önceden işlenmiş tensor'ları ver)
    class NewsDataset(Dataset):
        def __init__(self, encodings, labels):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    def encode(texts):
        return tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )

    print("[Train] Tokenization yapiliyor (train)...")
    train_enc = encode(train_texts)
    print("[Train] Tokenization yapiliyor (val)...")
    val_enc = encode(val_texts)
    print("[Train] Tokenization yapiliyor (test)...")
    test_enc = encode(test_texts)

    train_dataset = NewsDataset(train_enc, train_labels)
    val_dataset = NewsDataset(val_enc, val_labels)
    test_dataset = NewsDataset(test_enc, test_labels)

    # Windows'ta CUDA multiprocessing sorunlarını önlemek için num_workers=0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    # Dropout artir (overfitting onlemek icin)
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2

    # Optimizer ve Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.15),
        num_training_steps=total_steps,
    )

    # Sınıf ağırlıkları (dengesiz veri için)
    label_counts = np.bincount(train_labels, minlength=NUM_LABELS).astype(float)
    class_weights = 1.0 / (label_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_LABELS
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    # Eğitim döngüsü
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience = 2          # Early stopping: 2 epoch iyileşme olmazsa dur
    no_improve = 0

    print(f"\n[Train] Eğitim başlıyor — {epochs} epoch, lr={lr}, batch={batch_size}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()  # Windows'ta CUDA senkronizasyonu
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels_batch)

                val_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1%}"
        )

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"    -> En iyi model kaydedildi (val_acc: {val_acc:.1%})")
        else:
            no_improve += 1
            print(f"    -> Iyilesme yok ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"[Train] Early stopping — {epoch}. epoch'ta duruldu.")
                break

    print("=" * 70)

    # --- TEST ---
    print("\n[Train] Test seti değerlendirmesi:")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    print("\n" + classification_report(all_labels, all_preds, target_names=LABELS))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Eğitim meta verisi
    meta = {
        "base_model": base_model,
        "num_labels": NUM_LABELS,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts),
        "best_val_acc": round(best_val_acc, 4),
        "labels": LABELS,
        "trained_at": datetime.now().isoformat(),
        "hf_dataset": "ituperceptron/turkish-financial-sentiment-256k"
        if use_hf
        else None,
    }
    with open(os.path.join(MODEL_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Plot
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(history["train_loss"], label="Train", marker="o")
            ax1.plot(history["val_loss"], label="Val", marker="s")
            ax1.set_title("Loss")
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")

            ax2.plot(history["train_acc"], label="Train", marker="o")
            ax2.plot(history["val_acc"], label="Val", marker="s")
            ax2.set_title("Accuracy")
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")

            plt.suptitle(
                f"BERTurk Fine-tune ({NUM_LABELS}-class) — {len(train_texts)} samples, best val_acc={best_val_acc:.1%}"
            )
            plt.tight_layout()
            plot_path = os.path.join(MODEL_DIR, "training_curves.png")
            plt.savefig(plot_path, dpi=150)
            print(f"\n[Train] Grafik kaydedildi: {plot_path}")
            plt.show()
        except Exception as e:
            print(f"[Train] Plot hatası: {e}")

    print(f"\n[Train] Eğitim tamamlandı! Model: {MODEL_DIR}")
    print(f"  En iyi val_acc: {best_val_acc:.1%}")


def evaluate():
    """Kaydedilmiş modeli test verileriyle değerlendir."""
    from news.sentiment.berturk_model import BERTurkSentiment

    model = BERTurkSentiment()
    if not model.available:
        print("[Evaluate] Model bulunamadı. Önce --train çalıştırın.")
        return

    test_cases = [
        ("Şirket rekor kar açıkladı yatırımcılar memnun", "pozitif"),
        ("Bilançoda büyük zarar var hisseler sert düştü", "negatif"),
        ("Borsa güne yatay başladı beklentiler devam ediyor", "notr"),
        ("TCMB faiz kararını açıkladı beklenti dahilinde", "notr"),
        ("THY yeni uçak siparişi verdi filosunu genişletiyor", "pozitif"),
        ("Şirkete SPK soruşturması açıldı manipülasyon iddiaları", "negatif"),
        ("Dolar kuru yeni zirve yaptı ithalatçılar tedirgin", "negatif"),
        ("Bedelsiz sermaye artırımı kararı açıklandı", "pozitif"),
        ("Fabrika yangınında üretim durdu ciddi hasar var", "negatif"),
        ("Analistler hisseye güçlü al tavsiyesi verdi", "pozitif"),
    ]

    print("\n=== BERTurk Sentiment Değerlendirme (3-class) ===\n")
    correct = 0
    for text, expected in test_cases:
        result = model.predict(text)
        hit = "OK" if result["label"] == expected else "XX"
        bar = "#" * int(result["score"] / 5)
        print(
            f"  {hit} [{result['label']:>8}] (beklenen: {expected:>8}) "
            f"score={result['score']:5.1f} conf={result['confidence']:.0%} | {text[:50]}"
        )
        if result["label"] == expected:
            correct += 1

    print(
        f"\n  Test doğruluğu: {correct}/{len(test_cases)} ({correct / len(test_cases):.0%})"
    )


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTurk Sentiment Eğitim (3-class)")
    parser.add_argument("--train", action="store_true", help="Modeli eğit")
    parser.add_argument("--evaluate", action="store_true", help="Modeli değerlendir")
    parser.add_argument("--collect", action="store_true", help="Haber başlıkları topla")
    parser.add_argument(
        "--label", action="store_true", help="Gemini ile otomatik etiketle"
    )
    parser.add_argument("--hf", action="store_true", help="HuggingFace dataset kullan")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch sayısı")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--count", type=int, default=2000, help="Toplanacak/kullanılacak örnek sayısı"
    )
    parser.add_argument("--plot", action="store_true", help="Eğitim grafiği")
    args = parser.parse_args()

    if args.collect:
        collect_headlines(args.count, use_hf=args.hf)
    if args.label:
        auto_label_with_gemini()
    if args.train:
        train(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            plot=args.plot,
            use_hf=args.hf,
            hf_samples=args.count,
        )
    if args.evaluate:
        evaluate()

    if not any([args.train, args.evaluate, args.collect, args.label]):
        parser.print_help()
