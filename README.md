# 📸 Vesikalık Fotoğraf Otomatik Kontrol API'si / Passport Photo Auto-Checker API

Bu proje, vesikalık fotoğrafların belirli standartlara uygunluğunu kontrol eden FastAPI tabanlı bir web servisidir.  
This project is a FastAPI-based backend service that checks if a photo meets official ID/passport photo standards.

---

## 🧠 Özellikler / Features

- 👤 Yüz tespiti / Face detection (MediaPipe + OpenCV)  
- 📏 Boyut ve oran kontrolü / Size & ratio control  
- 🧱 Arka plan, kontrast ve renk analizi / Background, contrast & color analysis  
- 🔍 Bulanıklık ve ışık analizi / Blur & brightness check  
- 📆 EXIF tarih/rotasyon kontrolü / EXIF timestamp and rotation check  
- 🧢 Aksesuar tespiti (opsiyonel) / Accessory detection (optional)  
- 🌍 Çoklu dil desteği / Multilingual support (TR/EN)  
- 📝 Detaylı hata mesajları / Detailed error messages  
- 📊 Loglama ve istatistik desteği / Logging and statistics  
- 🔐 Rate limit koruması / Rate limiting protection  

---

## 🚀 Kurulum / Setup

1. Ortamı hazırla / Create virtual environment:  
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Gerekli paketleri yükle / Install dependencies:  
   ```
   pip install -r requirements.txt
   ```

3. Sunucuyu çalıştır / Start server:  
   ```
   uvicorn vesikalik:app --reload
   ```

4. API arayüzü / API Docs:  
   [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔁 Toplu Test Modülü (client.py) / Batch Test Module

`client.py`, belirli bir klasördeki tüm fotoğrafları API'ye göndererek topluca test etmenizi sağlar.  
`client.py` allows automated batch testing of multiple images via terminal.

- Fotoğraflar `Kontrol/` klasörüne konur / Place images into `Kontrol/`  
- Uygun olanlar `Uygun/`, diğerleri `Red/` klasörüne taşınır / Sorted accordingly  
- Tüm sonuçlar `client_logs.txt` dosyasına kaydedilir / Logged to `client_logs.txt`  
- Çok dilli sonuç seçeneği vardır / Supports `lang` param (`tr` / `en`)  
- Çalıştırmak için / Run with:  
   ```
   python client.py
   ```

---

## 🔗 API Endpoint'leri / API Endpoints

- **POST `/upload/`**  
  Fotoğrafı yükler ve uygunluk kontrolü yapar.  
  Uploads photo and checks compliance.

  Parametreler / Parameters:  
  - `file`: JPEG/PNG dosyası / image file  
  - `lang`: `tr` veya `en` (opsiyonel / optional)  

- **GET `/version`**  
  API sürümünü döndürür.  
  Returns current version.

---

## 💻 Gereksinimler / Requirements

- Python 3.9+  
- macOS (M serisi destekli) / M-series Macs supported  
- Windows/Linux uyumlu olabilir / Windows/Linux compatibility possible  

---

## 👩‍💻 Geliştirici / Developer

- **Ad:** Ömrüm Ceren GÜLER  
  **İletişim / Contact:** omrumguler35@gmail.com  
  **Staj:** Yapay Zeka & Yazılım Geliştirme / AI & Software Development  
  **Üniversite / University:** Çukurova Üniversitesi  
  **Yıl / Year:** 2025  
  **Koordinatör / Supervisor:** Mehmet Harun GÜLEN  

---

## 📝 Lisans / License

MIT License  
Tüm hakları geliştiriciye aittir. / All rights reserved by the developer.
