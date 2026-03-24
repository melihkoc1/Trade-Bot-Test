# Base image: Hafif bir Python surumu kullanilir
FROM python:3.10-slim

# Ortam degiskenleri
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Calisma dizinini ayarla
WORKDIR /app

# Gereksinimleri onceden kopyalayip cache avantajindan yararlan
COPY requirements.txt .

# Bagimliliklari yukle
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Proje dosyalarinin tamamini kopyala
COPY . .

# Streamlit varsayilan portu
EXPOSE 8501

# Uygulamayi baslatan komut (Streamlit baslatma)
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
