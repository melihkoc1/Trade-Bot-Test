"""
News Sentiment LSTM — Tradebot V1 (PyTorch)
============================================
Turkce finansal haber basliklarini siniflandiran BiLSTM modeli.

Siniflar:
  0 = NEGATIF (SAT sinyali)
  1 = NOTR    (bilgi amacli)
  2 = POZITIF (AL sinyali)

Mimari:
  Embedding(vocab_size, 64) -> Bidirectional LSTM(64) -> Dense(32, relu) -> Dense(3, softmax)

Kullanim:
  python sentiment_lstm.py --train [--plot]
  from sentiment_lstm import get_sentiment
"""

import os, re, json, numpy as np

MODEL_PATH = "models/sentiment_lstm.pt"
VOCAB_PATH = "models/sentiment_vocab.json"
MAX_LEN    = 30
VOCAB_SIZE = 5000
EMBED_DIM  = 64
LSTM_UNITS = 64
LABELS     = ["NEGATIF", "NOTR", "POZITIF"]

LABELED_DATA = [
    # POZITIF (2)
    ("sirket karini ikiye katladi",                              2),
    ("guclu buyume rakamlari aciklandi",                         2),
    ("hisse rekor kirdi",                                        2),
    ("temettu odemesi artirildi",                                2),
    ("beklentilerin uzerinde satis geliri",                      2),
    ("ihracat rekor seviyede",                                   2),
    ("analistler hedef fiyati yukseltti",                        2),
    ("sirket yeni fabrika yatirimi yapiyor",                     2),
    ("olumlu bilanco aciklandi",                                 2),
    ("borsa gune yukselisle basladi",                            2),
    ("yabanci yatirimci alimlari artiyor",                       2),
    ("faiz karari piyasalari rahatlatti",                        2),
    ("enflasyon beklentinin altinda geldi",                      2),
    ("buyume tahminleri yukari revize edildi",                   2),
    ("sirketten guclu nakit akisi",                              2),
    ("hisse senedi bolunmesi aciklandi",                         2),
    ("stratejik ortaklik anlasmasi imzalandi",                   2),
    ("uretim kapasitesi genisletiliyor",                         2),
    ("ihracat pazari buyudu",                                    2),
    ("sirkete uluslararasi kredi notu yukseldi",                 2),
    ("net kar beklentilerin uzerinde geldi",                     2),
    ("hisse senedi yeni zirveye ulasti",                         2),
    ("guclu satis hacmi aciklandi",                              2),
    ("sirket karlilık rekorunu kirdi",                           2),
    ("borsa tum zamanlarin en yuksek seviyesinde",               2),
    ("sirket yeni ihracat anlasmasi imzaladi",                   2),
    ("kar marjlari beklentilerin uzerinde",                      2),
    ("yatirimci guveni artti",                                   2),
    ("hisse deger kazanmaya devam ediyor",                       2),
    ("yeni yatirim tesviki alindi",                              2),
    ("buyuk yatirim paketi aciklandi",                           2),
    ("piyasa degeri rekor tazeledi",                             2),
    ("guclu talep artisi yasandi",                               2),
    ("borsa endeksi yuzde bes yukseldi",                         2),
    ("sirket gelirlerini yuzde otuz artirdi",                    2),
    ("kar payi dagitimi artti",                                  2),
    ("sirketten pozitif operasyonel sonuclar",                   2),
    ("hisseye guclu al tavsiyesi verildi",                       2),
    ("ciro rekor seviyeye ulasti",                               2),
    ("sirket yeni ortaklik kurdu",                               2),
    # NEGATIF (0)
    ("sirket zarar acikladi",                                    0),
    ("bilanco beklentilerin altinda kaldi",                      0),
    ("kur krizi piyasalari carpti",                              0),
    ("faiz artisi hisseleri dusurdu",                            0),
    ("enflasyon zirveye ulasti",                                 0),
    ("sirket konkordato ilan etti",                              0),
    ("hisse sert dustu",                                         0),
    ("yonetim kurulu istifa etti",                               0),
    ("sirkete vergi cezasi kesildi",                             0),
    ("uretim durdu",                                             0),
    ("ihracat yasaklandi",                                       0),
    ("borclar odenemiyor",                                       0),
    ("piyasalar cokuste",                                        0),
    ("doviz kurlari cildirdi",                                   0),
    ("buyume tahminleri asagi revize edildi",                    0),
    ("sirket yoneticisi gozaltina alindi",                       0),
    ("hisse devre kesici calisti",                               0),
    ("talep dususu yasandi",                                     0),
    ("sirkete sorusturma acildi",                                0),
    ("hisse yuzde on deger kaybetti",                            0),
    ("borsa sert geriledi",                                      0),
    ("sirketin kredisi donduruldu",                              0),
    ("iflas basvurusu yapildi",                                  0),
    ("kar beklentilerin cok altinda kaldi",                      0),
    ("sirkete para cezasi kesildi",                              0),
    ("calisan sayisi azaltiliyor",                               0),
    ("fabrika kapatma karari alindi",                            0),
    ("hisseler coktu yatirimcilar panikliyor",                   0),
    ("sirket zarar etmeye devam ediyor",                         0),
    ("borsa panikle satis gordu",                                0),
    ("kredi notu dusuruldu",                                     0),
    ("sirketten kotu haberler geliyor",                          0),
    ("sektorde buyuk kriz bas gosterdi",                         0),
    ("doviz kuru zirve yapti sirket zarari buyudu",              0),
    ("yuksek borc yuku altinda ezilen sirket",                   0),
    ("hisse senetleri coktu",                                    0),
    ("sirkette ciddi yonetim krizi yasaniyor",                   0),
    ("piyasalarda guven krizi derinlesiyor",                     0),
    ("sirket satislarini durdurmak zorunda kaldi",               0),
    # NOTR (1)
    ("sirket yonetim kurulu toplandi",                           1),
    ("faiz karari aciklandi",                                    1),
    ("borsa gunu yatay kapatti",                                 1),
    ("beklentiler degismedi",                                    1),
    ("sirket genel kurul tarihi belirlendi",                     1),
    ("piyasalar temkinli seyrediyor",                            1),
    ("analistler beklenti degisikligi yapmadi",                  1),
    ("sirket yonetim degisikligi yapti",                         1),
    ("sektor raporu yayimlandi",                                 1),
    ("hisse hacmi normale dondu",                                1),
    ("sirket yeni urun duyurdu",                                 1),
    ("piyasalar veri beklentisiyle sakin",                       1),
    ("merkez bankasi toplantisi yaklasiyor",                     1),
    ("sirket bagimsiz denetim surecini tamamladi",               1),
    ("borsa endeksi yatay seyretti",                             1),
    ("sirket yillik raporu yayimladi",                           1),
    ("piyasalar dolar kuru takibinde",                           1),
    ("analistler bekle tavsiyesini korudu",                      1),
    ("sirket aciklama yapacak",                                  1),
    ("borsa gunluk islem hacmi ortalama seyretti",               1),
    ("sirketin ceyrek sonuclari bekleniyor",                     1),
    ("piyasalar abd verilerini bekliyor",                        1),
    ("sirket yeni ceo atadi",                                    1),
    ("analistler sirketi izlemeye aldi",                         1),
    ("borsa haftayi yatay kapatti",                              1),
    ("sirket faaliyet raporunu yayimladi",                       1),
    ("piyasalar merkez bankasi kararini bekliyor",               1),
    ("hisse bugun cok az islem gordu",                           1),
    ("sirket stratejik plan sundu",                              1),
    ("borsa endeksinde sinirli hareketler",                      1),
    ("sirketin temettu politikasi degismedi",                    1),
    ("piyasalar secim oncesi bekleme modunda",                   1),
    ("sirket bilgi guncelleme toplantisi yapti",                 1),
    ("analistler notr gorusunu korudu",                          1),
    ("borsa gunluk bazda degismedi",                             1),
]


# ──────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()


def build_vocab(texts: list) -> dict:
    from collections import Counter
    words = []
    for t in texts:
        words.extend(_normalize(t).split())
    counts = Counter(words)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counts.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts: list, vocab: dict) -> np.ndarray:
    seqs = []
    for t in texts:
        tokens = [vocab.get(w, 1) for w in _normalize(t).split()]
        tokens = tokens[:MAX_LEN]
        tokens += [0] * (MAX_LEN - len(tokens))
        seqs.append(tokens)
    return np.array(seqs, dtype=np.int64)


# ──────────────────────────────────────────────────────────────
# PyTorch Model
# ──────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, LSTM_UNITS, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.fc1  = nn.Linear(LSTM_UNITS * 2, 32)
        self.drop = nn.Dropout(0.5)
        self.fc2  = nn.Linear(32, 3)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h = torch.relu(self.fc1(h))
        h = self.drop(h)
        return self.fc2(h)


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_net() -> _Net:
    return _Net().to(_DEVICE)


# ──────────────────────────────────────────────────────────────
# Egitim
# ──────────────────────────────────────────────────────────────

def train_model(extra_data: list = None, epochs: int = 150, plot: bool = False):
    data = list(LABELED_DATA)
    if extra_data:
        data.extend(extra_data)

    texts  = [d[0] for d in data]
    labels = np.array([d[1] for d in data], dtype=np.int64)

    vocab = build_vocab(texts)
    X = texts_to_sequences(texts, vocab)

    idx = np.random.permutation(len(X))
    X, labels = X[idx], labels[idx]

    split = max(1, int(0.8 * len(X)))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = labels[:split], labels[split:]

    net = _make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    X_tr = torch.tensor(X_train).to(_DEVICE)
    y_tr = torch.tensor(y_train).to(_DEVICE)
    X_vl = torch.tensor(X_val).to(_DEVICE)
    y_vl = torch.tensor(y_val).to(_DEVICE)

    best_val_acc = 0.0
    best_state   = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"[SentimentLSTM] Egitim basliyor — {len(X_train)} train, {len(X_val)} val | cihaz: {_DEVICE}")

    for epoch in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        loss = crit(net(X_tr), y_tr)
        loss.backward()
        opt.step()
        sch.step()

        net.eval()
        with torch.no_grad():
            tr_acc = (net(X_tr).argmax(1) == y_tr).float().mean().item()
            vl_out = net(X_vl)
            vl_loss = crit(vl_out, y_vl).item()
            vl_acc  = (vl_out.argmax(1) == y_vl).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {loss.item():.4f} acc {tr_acc:.0%} | "
                  f"val_loss {vl_loss:.4f} val_acc {vl_acc:.0%}")

    net.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[SentimentLSTM] Model kaydedildi. En iyi val_acc: {best_val_acc:.0%}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
            ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
            ax2.plot(history["train_acc"], label="Train"); ax2.plot(history["val_acc"], label="Val")
            ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True)
            plt.tight_layout()
            plt.savefig("models/sentiment_training.png", dpi=120)
            print("[SentimentLSTM] Grafik: models/sentiment_training.png")
            plt.show()
        except Exception as e:
            print(f"Grafik hatasi: {e}")

    return history


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

_net_cache:   _Net  = None
_vocab_cache: dict  = None


def _load() -> bool:
    global _net_cache, _vocab_cache
    if _net_cache is not None:
        return True
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        return False
    try:
        net = _make_net()
        net.load_state_dict(torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=True))
        net.eval()
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab = json.load(f)
        _net_cache   = net
        _vocab_cache = vocab
        return True
    except Exception as e:
        print(f"[SentimentLSTM] Model yuklenemedi: {e}")
        return False


def get_sentiment(text: str) -> tuple:
    """Tek metin -> (label, confidence). Ornek: ('POZITIF', 0.87)"""
    if not _load():
        return "NOTR", 0.0
    try:
        X = torch.tensor(texts_to_sequences([text], _vocab_cache)).to(_DEVICE)
        with torch.no_grad():
            probs = torch.softmax(_net_cache(X), dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])
    except Exception as e:
        print(f"[SentimentLSTM] Tahmin hatasi: {e}")
        return "NOTR", 0.0


def batch_sentiment(texts: list) -> list:
    """Toplu tahmin -> [(label, confidence), ...]"""
    if not _load():
        return [("NOTR", 0.0)] * len(texts)
    try:
        X = torch.tensor(texts_to_sequences(texts, _vocab_cache)).to(_DEVICE)
        with torch.no_grad():
            probs = torch.softmax(_net_cache(X), dim=1).cpu().numpy()
        return [(LABELS[int(np.argmax(p))], float(np.max(p))) for p in probs]
    except Exception as e:
        print(f"[SentimentLSTM] Batch hatasi: {e}")
        return [("NOTR", 0.0)] * len(texts)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    if args.train:
        train_model(epochs=args.epochs, plot=args.plot)

    if args.test or not args.train:
        print("\n--- Ornek Tahminler ---")
        tests = [
            "sirket karini ikiye katladi",
            "bilanco beklentilerin altinda kaldi",
            "borsa gunu yatay kapatti",
            "Garanti Bankasi rekor kar acikladi",
            "fabrika yangini nedeniyle uretim durdu",
            "sirket yonetim kurulu toplandi",
        ]
        for t in tests:
            label, conf = get_sentiment(t)
            bar = ">" * int(conf * 20)
            print(f"  {label:8s} {conf:.0%} {bar:20s} | {t}")
