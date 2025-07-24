## 📄 Türkçe Version (README)

# Vesikalık Fotoğraf Otomatik Kontrol API'si

Bu proje, yüklenen bir fotoğrafın vesikalık standartlarına uygun olup olmadığını otomatik olarak kontrol eden FastAPI tabanlı bir backend servisidir.

## 🚀 Özellikler

- **Yüz tespiti:** MediaPipe ve OpenCV ile otomatik yüz algılama
- **Boyut ve oran kontrolü:** En-boy oranı, çözünürlük sınırları
- **Arka plan, netlik ve renk analizi**
- **Bulanıklık ve aydınlık/karanlık kontrolü**
- **EXIF tarih ve rotasyon analizi**
- **Boydan fotoğraf tespiti**
- **Gözlük, aksesuar vb. kontrol (isteğe bağlı)**
- **Çoklu dil desteği (Türkçe/İngilizce)**
- **Ayrıntılı hata mesajları**
- **Log ve hata takibi**
- **Versiyon endpoint'i**  
- **Rate limit koruması** (sistemi korumak için istek sınırlama)

## 🛠️ Kurulum

1. Sanal ortamı etkinleştir (önerilir):
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Gerekli kütüphaneleri yükle:
    ```
    pip install -r requirements.txt
    ```
    veya (eğer requirements yoksa):
    ```
    pip install fastapi uvicorn opencv-python mediapipe pillow numpy filetype slowapi python-multipart
    ```
3. Sunucuyu başlat:
    ```
    uvicorn vesikalik:app --reload
    ```
4. API dokümantasyonuna git:  
   [http://localhost:8000/docs](http://localhost:8000/docs)

## ⚡️ API Endpointleri

- **POST `/upload/`**  
  Vesikalık fotoğrafı yükle ve otomatik kontrole sokar.  
  Parametreler:  
    - `file`: Fotoğraf dosyası (JPEG/PNG)
    - `lang`: tr/en (isteğe bağlı, varsayılan Türkçe)

- **GET `/version`**  
  API sürüm ve build tarihi bilgisini döndürür.

## 💡 Kullanım Notları

- Fotoğraf vesikalık kurallarına uygun değilse ayrıntılı, çok dilli hata mesajı döner.
- Her başarılı yüklemede dosyanın orijinal adı, boyutu ve yükleme zamanı JSON olarak gelir.
- Log dosyası otomatik oluşturulur (`vesikalik_logs.txt`).
- Rate limit sistemi ile ardışık istekler engellenir.

## 🔧 Gereksinimler

- Python 3.9 veya üzeri
- MacOS (M1/M2/M3/M4 uyumlu test edildi)
- (Windows/Linux desteği de mümkündür)

## 👩‍💻 Geliştirici & Proje Bilgisi

- **Adı:** Ömrüm Ceren GÜLER
- **İletişim:** omrumguler35@gmail.com
- **Staj:** Yapay Zeka & Yazılım Geliştirme
- **Yıl:** 2025
- **Üniversite:** Çukurova Üniversitesi
- **Staj Koordinatörü:** Mehmet Harun GÜLEN

## Lisans

MIT  
(Tüm telif ve kullanım hakları geliştiriciye aittir.)

---

**Not:**  
Bu dosya, API'nin temel kullanımını ve geliştirme amaçlarını açıklar.  
Güncellenmesi veya özelleştirilmesi gerekirse lütfen geliştirici ile iletişime geçiniz.

---

---

## 📄 English Version (README)

# Automatic Passport/ID Photo Checker API

This project is a FastAPI-based backend service that automatically checks if an uploaded photo meets the standard requirements for official passport or ID photos.

## 🚀 Features

- **Face detection:** Automatic face detection using MediaPipe and OpenCV
- **Size and aspect ratio check:** Dimension and ratio controls
- **Background, sharpness, and color analysis**
- **Blurriness and brightness/darkness checks**
- **EXIF date and orientation analysis**
- **Full-body photo detection**
- **Accessory detection (optional, e.g., glasses, hats)**
- **Multi-language support (Turkish/English)**
- **Detailed error messages**
- **Log and error tracking**
- **Version endpoint**
- **Rate limit protection** (prevents abuse)

## 🛠️ Setup

1. **Create and activate a virtual environment** (recommended):
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
2. **Install required libraries:**
    ```
    pip install -r requirements.txt
    ```
    Or, if there is no requirements file:
    ```
    pip install fastapi uvicorn opencv-python mediapipe pillow numpy filetype slowapi python-multipart
    ```
3. **Start the server:**
    ```
    uvicorn vesikalik:app --reload
    ```
4. **Open the API documentation:**  
   [http://localhost:8000/docs](http://localhost:8000/docs)

## ⚡️ API Endpoints

- **POST `/upload/`**  
  Upload a passport/ID photo and run the automatic compliance check.  
  Parameters:  
    - `file`: Image file (JPEG/PNG)
    - `lang`: tr/en (optional, default is Turkish)

- **GET `/version`**  
  Returns API version and build date info.

## 💡 Usage Notes

- If the photo does not meet the requirements, you will get detailed and multilingual error messages.
- Each successful upload returns the original filename, file size, and upload time as JSON.
- Log file is generated automatically (`vesikalik_logs.txt`).
- Rate limiting is enabled to prevent abuse.

## 🔧 Requirements

- Python 3.9 or later
- MacOS (tested on M1/M2/M3/M4)
- (Windows/Linux support is possible)

## 👩‍💻 Developer & Project Info

- **Name:** Ömrüm Ceren GÜLER
- **Contact:** omrumguler35@gmail.com
- **Internship:** AI & Software Development
- **Year:** 2025
- **University:** Çukurova University
- **Internship Coordinator:** Mehmet Harun GÜLEN

## License

MIT  
(All copyright and usage rights belong to the developer.)

---

**Note:**  
This file explains the main usage and development goals of the API.  
If you need to update or customize it, please contact the developer.

---
