---
name: phase
description: Tradebot V1 için yeni bir geliştirme fazı planla ve uygula. Kullanıcı yeni bir özellik, değişiklik veya iyileştirme yapmak istediğinde tetikle.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Edit, Write
---

# Tradebot V1 — Yeni Faz Uygulama Süreci

Görev: **$ARGUMENTS**

## Adım 1: Mevcut Durumu Anla

Şu dosyaları oku ve projenin mevcut durumunu kavra:
- `strategy.py` — Aktif kurallar ve skorlama sistemi
- `indicators.py` — Hesaplanan indikatörler
- `backtester.py` — Backtest motoru

## Adım 2: Etki Analizi

Yapılacak değişikliğin şu sorulara cevap ver:

1. **Hangi dosyalar etkilenecek?** (strategy.py, indicators.py, app.py, backtester.py?)
2. **Look-ahead bias riski var mı?** — Eğer strateji veya indikatör değişiyorsa, backtest'te `is_backtest=True` guard'ı gerekiyor mu?
3. **Yeni bir indikatör ekleniyorsa** — `add_technical_indicators()` içine eklenmeli
4. **Yeni bir kural ekleniyorsa** — `if not is_backtest:` guard'ı gerekiyor mu?

## Adım 3: Uygulama Planı

Kullanıcıya şu formatı yaz:

```
PLAN:
- Değiştirilecek dosyalar: [liste]
- Yeni kolonlar/fonksiyonlar: [liste]
- is_backtest guard gerekiyor mu: [Evet/Hayır — neden]
- Tahmini etki: [Skor üzerindeki max puan etkisi]
```

## Adım 4: Uygula

Kullanıcı onaylarsa değişiklikleri yap:
- Yeni indikatörler için `indicators.py`'ı güncelle
- Yeni kurallar için `strategy.py`'a ekle, `is_backtest` guard'larını koy
- `result` dict'ine yeni alan ekliyorsan hem başlangıç değerini hem de doldurma kısmını yaz

## Önemli Kurallar

- Backtest'te dış API çağrısı yapma (fundamental, news, weekly fetch hepsi `if not is_backtest:` arkasında olmalı)
- Yeni ML modeli ekleniyorsa backtest'te tamamen atla
- Skor katkısı net olarak belirt (max ±kaç puan?)
- `result` dict'inde kullanılmayan legacy alanları temizle