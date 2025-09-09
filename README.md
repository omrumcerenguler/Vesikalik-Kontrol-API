# Vesikalık Fotoğraf Otomatik Kontrol Sistemi

## Proje Açıklaması
Bu proje, vesikalık (kimlik/pasaport) fotoğrafların resmi standartlara uygunluğunu otomatik olarak değerlendiren bir API ve toplu kontrol aracıdır. FastAPI tabanlı arka uç servis, çeşitli görüntü işleme teknikleriyle fotoğrafları analiz eder ve ayrıntılı uygunluk raporları sunar. Ayrıca, toplu test için komut satırı istemcisi (client.py) ile birlikte gelir.

---

## Özellikler
- Yüz tespiti (MediaPipe, OpenCV)
- Fotoğraf boyutu, çözünürlük ve oran kontrolü
- Arka plan, kontrast ve renk analizi
- Bulanıklık ve ışık/aydınlatma denetimi
- EXIF veri ve rotasyon kontrolü
- Aksesuar (gözlük, şapka vb.) tespiti (isteğe bağlı)
- Çoklu dil desteği (Türkçe/İngilizce)
- Detaylı hata ve açıklama mesajları
- Loglama ve istatistik toplama
- Rate limit (istek kısıtlama) koruması

---

## Kurulum
1. Sanal ortam oluşturun:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
   ```
   # Windows
   venv\Scripts\activate

   # Linux
   source venv/bin/activate
   ```
2. Gereksinimleri yükleyin:
   ```
   pip install -r requirements.txt
   ```

---

## Çalıştırma Adımları
### API Sunucusu
1. Sunucuyu başlatın:
   ```
   uvicorn vesikalik:app --reload
   ```
2. Dokümantasyon arayüzü: [http://localhost:8000/docs](http://localhost:8000/docs)

### Toplu Test Aracı (client.py)
1. Kontrol edilecek fotoğrafları `Kontrol/` klasörüne yerleştirin.
2. Terminalden çalıştırın:
   ```
   python client.py
   ```
3. Sonuçlar, uygun fotoğraflar için `Uygun/`, reddedilenler için `Red/` klasörlerine ve ayrıca `client_logs.txt` dosyasına kaydedilir.
4. `lang` parametresi ile dil seçebilirsiniz (`tr` veya `en`).

---

## API Endpoint'leri
- **POST `/upload/`**  
  Fotoğraf dosyasını alır ve uygunluk kontrolü yapar.
  - Parametreler:
    - `file`: JPEG veya PNG formatında fotoğraf dosyası
    - `lang`: `tr` veya `en` (opsiyonel)

- **GET `/version`**  
  API sürüm bilgisini döndürür.

---

## Gereksinimler
- Python 3.9 veya üzeri
- macOS (M serisi desteklenir), Windows ve Linux ile uyumlu

---

## Geliştirici Bilgisi
- **İsim:** Ömrüm Ceren GÜLER
- **E-posta:** omrumguler35@gmail.com
- **Staj:** Yapay Zeka & Yazılım Geliştirme
- **Üniversite:** Çukurova Üniversitesi
- **Yıl:** 2025
- **Koordinatör:** Mehmet Harun GÜLEN

---


# Passport Photo Auto-Checker System

## Project Description
This project provides an API and a batch control tool for automatically checking official ID/passport photos for compliance with standards. The FastAPI-based backend analyzes images using computer vision techniques and produces detailed compliance reports. A command-line client (client.py) is included for batch testing.

---

## Features
- Face detection (MediaPipe, OpenCV)
- Photo size, resolution, and aspect ratio control
- Background, contrast, and color analysis
- Blur and brightness/illumination check
- EXIF data and rotation check
- Accessory (glasses, hats, etc.) detection (optional)
- Multi-language support (Turkish/English)
- Detailed error and explanation messages
- Logging and statistics
- Rate limiting protection

---

## Installation
1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
   ```
   # Windows
   venv\Scripts\activate

   # Linux
   source venv/bin/activate
   ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

---

## Usage
### API Server
1. Start the server:
   ```
   uvicorn vesikalik:app --reload
   ```
2. Documentation interface: [http://localhost:8000/docs](http://localhost:8000/docs)

### Batch Test Tool (client.py)
1. Place images to be checked into the `Kontrol/` folder.
2. Run via terminal:
   ```
   python client.py
   ```
3. Results are saved: compliant photos to `Uygun/`, rejected ones to `Red/`, and all logs to `client_logs.txt`.
4. Use the `lang` parameter for language selection (`tr` or `en`).

---

## API Endpoints
- **POST `/upload/`**  
  Accepts an image file and checks for compliance.
  - Parameters:
    - `file`: Photo file in JPEG or PNG format
    - `lang`: `tr` or `en` (optional)

- **GET `/version`**  
  Returns the API version information.

---

## Requirements
- Python 3.9 or higher
- macOS (M-series supported), compatible with Windows and Linux

---

## Developer Info
- **Name:** Ömrüm Ceren GÜLER
- **Email:** omrumguler35@gmail.com
- **Internship:** AI & Software Development
- **University:** Çukurova University
- **Year:** 2025
- **Supervisor:** Mehmet Harun GÜLEN

---
