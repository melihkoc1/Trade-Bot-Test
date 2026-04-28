"""
Tradebot V1 — Finansal Duygu Analizi Demo
==========================================
Cümle gir, NLP + Makro etki analizini gör.

Çalıştırmak için:
    streamlit run demo_sentiment.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from news.macro_impact import MacroImpactCalculator

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finansal Duygu Analizi",
    page_icon="📰",
    layout="centered",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "berturk_sentiment")
LABELS = ["negatif", "notr", "pozitif"]
LABEL_EMOJIS = {"negatif": "🔴", "notr": "🟡", "pozitif": "🟢"}
LABEL_COLORS = {"negatif": "#ff4b4b", "notr": "#ffa500", "pozitif": "#00c853"}

# ─── Model Yükle (cache) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_macro():
    return MacroImpactCalculator()

tokenizer, model, device = load_model()
calc = load_macro()

# ─── NLP Tahmin ───────────────────────────────────────────────────────────────
def predict(text: str):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return LABELS[pred], probs.cpu().tolist()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📰 Finansal Duygu Analizi")
st.caption("BERTurk tabanlı Türkçe finansal haber sınıflandırıcı + Makro etki motoru")

st.markdown("---")

text = st.text_area(
    "Haber başlığı veya metni girin:",
    placeholder="Örnek: Merkez bankası 2.5 baz puan faiz indirimine gitti",
    height=80,
)

col1, col2 = st.columns([1, 5])
with col1:
    analiz_btn = st.button("Analiz Et", type="primary", use_container_width=True)

if analiz_btn and text.strip():
    label, probs = predict(text.strip())
    makro = calc.analyze_news_text(text.strip())

    # ── NLP Sonucu ──────────────────────────────────────────────────────────
    st.markdown("### NLP Duygu Analizi")

    emoji = LABEL_EMOJIS[label]
    color = LABEL_COLORS[label]
    st.markdown(
        f"<h2 style='color:{color};'>{emoji} {label.upper()}</h2>",
        unsafe_allow_html=True,
    )

    # Olasılık barları
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🔴 Negatif", f"{probs[0]:.1%}")
        st.progress(probs[0])
    with c2:
        st.metric("🟡 Nötr", f"{probs[1]:.1%}")
        st.progress(probs[1])
    with c3:
        st.metric("🟢 Pozitif", f"{probs[2]:.1%}")
        st.progress(probs[2])

    st.markdown("---")

    # ── Makro Analiz ────────────────────────────────────────────────────────
    st.markdown("### Makro Etki Analizi")

    if makro["category"]:
        yon_emoji = {"artis": "📈", "indirim": "📉"}.get(makro["direction"], "➡️")
        st.info(
            f"**Kategori:** {makro['category'].capitalize()}  |  "
            f"**Yön:** {yon_emoji} {makro['direction'] or 'belirsiz'}"
        )

        if makro["sector_impacts"]:
            st.markdown("**Sektör Bazlı Etki:**")
            sorted_impacts = sorted(
                makro["sector_impacts"].items(), key=lambda x: -abs(x[1])
            )
            for sektor, skor in sorted_impacts:
                bar_color = "🟢" if skor > 0 else "🔴"
                isaretli = f"+{skor:.3f}" if skor > 0 else f"{skor:.3f}"
                st.markdown(
                    f"{bar_color} **{sektor}**: `{isaretli}`"
                )
    else:
        st.warning(
            "Bu haber makro kategori içermiyor (şirket haberi olabilir). "
            "NLP skoru yeterli."
        )

    st.markdown("---")

    # ── Açıklama ────────────────────────────────────────────────────────────
    with st.expander("ℹ️ Bu sonuçlar nasıl hesaplanıyor?"):
        st.markdown(
            """
**NLP Modeli:** `savasy/bert-base-turkish-sentiment-cased` üzerine
100.000 Türkçe finansal haber ile fine-tune edilmiş BERTurk modeli.
Val accuracy: **%84.3**, Test accuracy: **%84**.

**Makro Etki Motoru:** Kural tabanlı sistem.
Haberin konusunu (faiz, enflasyon, kur, petrol, altın) ve yönünü
(artış/indirim) tespit ederek önceden tanımlanmış ekonomik
korelasyon tablosundan sektör etkilerini hesaplar.

> Örnek: Faiz indirimi → GYO, İnşaat, Banka sektörlerine olumlu etki
            """
        )

elif analiz_btn and not text.strip():
    st.warning("Lütfen bir metin girin.")

# ─── Örnek Haberler ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Örnek Haberler")

ornekler = [
    "Merkez bankası 2.5 baz puan faiz indirimine gitti",
    "TCMB politika faizini 250 baz puan artırdı",
    "Şirket rekor kâr açıkladı, hisseler tavan yaptı",
    "Enflasyon beklentinin üzerinde çıktı, yüzde 65 olarak açıklandı",
    "Dolar zirve yaptı, TL değer kaybetti",
    "Şirket zarar açıkladı, hisseler sert düştü",
    "Piyasalar belirsiz seyrediyor, yatırımcılar bekliyor",
]

cols = st.columns(2)
for i, ornek in enumerate(ornekler):
    with cols[i % 2]:
        if st.button(ornek, key=f"ornek_{i}", use_container_width=True):
            st.session_state["ornek_text"] = ornek
            st.rerun()

# Örnek seçildiyse text_area'ya yaz
if "ornek_text" in st.session_state:
    selected = st.session_state.pop("ornek_text")
    label, probs = predict(selected)
    makro = calc.analyze_news_text(selected)

    st.markdown(f"**Seçilen:** `{selected}`")

    emoji = LABEL_EMOJIS[label]
    color = LABEL_COLORS[label]
    st.markdown(
        f"<h3 style='color:{color};'>{emoji} {label.upper()} "
        f"(güven: {max(probs):.1%})</h3>",
        unsafe_allow_html=True,
    )

    if makro["category"]:
        yon_emoji = {"artis": "📈", "indirim": "📉"}.get(makro["direction"], "➡️")
        st.info(
            f"**Makro:** {makro['category'].capitalize()} | "
            f"{yon_emoji} {makro['direction'] or 'belirsiz'}"
        )
        if makro["sector_impacts"]:
            for sektor, skor in sorted(
                makro["sector_impacts"].items(), key=lambda x: -abs(x[1])
            ):
                bar_color = "🟢" if skor > 0 else "🔴"
                isaretli = f"+{skor:.3f}" if skor > 0 else f"{skor:.3f}"
                st.markdown(f"{bar_color} **{sektor}**: `{isaretli}`")
