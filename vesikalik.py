from fastapi import FastAPI, File, UploadFile           # FastAPI ana paketleri, dosya upload için
from fastapi.responses import JSONResponse              # Kullanıcıya düzgün JSON mesajı döndürmek için
import shutil                                           # Dosya kopyalamak için
import os                                               # Dosya/dizin işlemleri için
import cv2                                              # OpenCV: Resim işleme ve okuma
import mediapipe as mp                                  # MediaPipe: Yüz tespiti için ML tabanlı kütüphane
import numpy as np                                      # NumPy: Matematiksel işlemler ve dizi manipülasyonu
import filetype                                         # Dosya türünü tahmin etmek için
from PIL import Image, ExifTags                         # PIL: Resim işleme ve EXIF verilerini okumak için
import datetime                                         # Tarih ve saat işlemleri için
import uuid                                             # Benzersiz ID'ler oluşturmak için
import re                                               # Regex: Dosya adında Türkçe karakter ve özel karakter kontrolü için
from pathlib import Path                                # Path işlemleri için
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler # SlowAPI: Rate limiting için
from slowapi.util import get_remote_address # SlowAPI: İsteklerin IP adresini almak için  
from fastapi import Request # FastAPI istek nesnesi için
from slowapi.errors import RateLimitExceeded # Rate limit aşımı hatası için

# Rate limiter ayarı: IP adresine göre sınır koyar
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Basit log dosyası ayarı (gelişmişini istersek farklı isim, klasör, tarihli dosya ekleyebilirsin)
logging.basicConfig(
    filename="vesikalik_logs.txt",    # Loglar bu dosyada birikir
    level=logging.INFO,               # INFO ve üstü mesajları yaz
    format="%(asctime)s | %(levelname)s | %(message)s"
)
# --- FastAPI uygulaması başlatma ---
app = FastAPI()                                         # FastAPI uygulamasını başlat

# --- Sadece test etmek için (zorunlu değil), upload işleviyle ilgisiz ---
@app.post("/kontrol")
async def kontrol(file: UploadFile = File(...)):
    safe_filename = Path(file.filename).name  # Sadece dosya adını al, path silinir
    return {"filename": safe_filename, "message": "Fotoğraf alındı!"}

# --- Ana yüz tespit fonksiyonu: Fotoğrafta kaç yüz var, ilk yüzün oranları ve konumu nedir? ---
def detect_face(image_path):
    mp_face = mp.solutions.face_detection                       # MediaPipe yüz tespit modülünü çağır
    img = cv2.imread(image_path)                                # Dosyayı OpenCV ile oku (resim olarak)
    if img is None:
        return -1, None, 0                                     # Resim açılamadıysa: hata kodu dön
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # MediaPipe için resmi RGB'ye çevir

    # MediaPipe ile yüzleri tespit et
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(img_rgb)                # Tespit edilen yüzlerin listesi
        if result.detections:
            num_faces = len(result.detections)                  # Kaç yüz bulundu?
            detection = result.detections[0]                    # İlk yüzü al (en büyük/genel olarak)
            bboxC = detection.location_data.relative_bounding_box# Yüzün oranlara göre bounding box'u
            h, w, _ = img.shape                                 # Fotoğrafın yüksekliği, genişliği
            box_w, box_h = bboxC.width * w, bboxC.height * h    # Yüzün genişliği ve yüksekliği (piksel cinsinden)
            face_area = box_w * box_h                           # Yüzün kapladığı toplam alan (piksel)
            img_area = w * h                                    # Tüm fotoğrafın alanı
            face_ratio = face_area / img_area                   # Yüzün fotoğrafa oranı (% olarak, 0.2 = %20)
            # Yüzün merkezi nerde? (x,y)
            face_center_x = bboxC.xmin * w + box_w / 2
            face_center_y = bboxC.ymin * h + box_h / 2
            # Fotoğrafın merkezi nerede?
            center_x, center_y = w / 2, h / 2
            # Yüzün merkezi ile fotoğraf merkezi arasındaki fark (0 ise tam ortada)
            offset_x = abs(face_center_x - center_x) / w        # Yatay uzaklık oranı
            offset_y = abs(face_center_y - center_y) / h        # Dikey uzaklık oranı
            # Dönülen veri: yüz bulundu kodu (1), yüz detayları, kaç yüz olduğu
            return 1, {"face_ratio": face_ratio, "offset_x": offset_x, "offset_y": offset_y}, num_faces
        else:
            # Hiç yüz bulunmadı
            return 0, None, 0

# --- ANA API: Fotoğrafı upload et, tüm vesikalık kontrollerini sırayla yap ---
@app.post("/upload/")
@limiter.limit("30/minute")   # ← Her IP dakikada 30 yükleme hakkı
async def upload_image(request: Request, file: UploadFile = File(...)):
    errors = []
    
    # --- Benzersiz ve güvenli dosya adı oluştur ---
    original_filename = Path(file.filename).name  # Sadece dosya adını al, path'i sil
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{unique_id}_{original_filename}"
   
    # --- KURAL 1: Dosya adı kontrolleri ---
    # Açıklama: Dosya adında Türkçe karakter, boşluk, çok uzun isim, özel karakter var mı?
    # 1.1 Türkçe karakter kontrolü
    if re.search(r'[çÇğĞıİöÖşŞüÜ]', filename):
        errors.append("Dosya adında Türkçe karakter kullanılamaz. Lütfen İngilizce karakterlerle adlandırınız.")

    # 1.2 Boşluk kontrolü
    if " " in filename:
        errors.append("Dosya adında boşluk bulunamaz. Lütfen boşluk yerine '_' veya '-' kullanınız.")

    # 1.3 Çok uzun dosya adı (ör: >100 karakter)
    if len(filename) > 100:
        errors.append(f"Dosya adı çok uzun ({len(filename)} karakter). Lütfen daha kısa bir dosya adı giriniz.")

    # 1.4 Uygunsuz özel karakterler
    if re.search(r'[\\/:*?"<>|]', filename):
        errors.append("Dosya adında uygunsuz karakterler var (\\ / : * ? \" < > |). Lütfen bu karakterleri kullanmayın.")

    # --- KURAL 2: Dosya tipi ve MIME-Type kontrolü ---
    # Dosyanın gerçekten resim olup olmadığını (JPEG/PNG) kontrol et
    file.file.seek(0)  # Dosyanın başına dön
    kind = filetype.guess(file.file.read(261))  #filetype.guess() dosyanın içinden türünü tahmin eder.
    file.file.seek(0)  # Okuma sonrası başa dön
    if not kind or kind.mime not in ['image/jpeg', 'image/png']:
        errors.append("Yüklenen dosya gerçek bir JPEG/PNG görsel değil. Lütfen sadece resim yükleyin.")
        return JSONResponse({"status": "error", "errors": errors})

    # --- KURAL 3: Dosya uzantısı kontrolü ---
    ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
    ext = filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        errors.append(f"Dosya formatı uygun değil: {ext}. Sadece jpg, jpeg, png yükleyebilirsiniz.")
    # 3.1 Dosya adında uzantı yoksa
    if '.' not in filename:
        errors.append("Dosya adında uzantı yok. Lütfen dosya adını '.jpg' veya '.png' ile bitirin.")
    
    # --- KURAL 4: Dosya uzantısı ve içeriği uyuşmazlık kontrolü ---
    # Yüklenen dosyanın uzantısı (jpg/png) ile gerçek dosya tipi aynı mı?
    mime_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    if ext in mime_map and kind and kind.mime != mime_map[ext]:
        errors.append(f"Dosya uzantısı ({ext}) ile dosyanın gerçek tipi (MIME: {kind.mime}) uyuşmuyor. Lütfen dosyanın adını değiştirmeden yükleyin.")

    # --- KURAL 5: uploads klasörü yoksa oluştur ---
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_location = f"uploads/{filename}" # Dosya kaydedilecek konum

    # --- KURAL 7: EXIF meta verisi ile tarih ve rotasyon kontrolü ---
    img_pil, exif_errors = correct_orientation_and_check_date(file)
    if exif_errors:
        errors.extend(exif_errors)

    # --- KURAL 6: Dosya boyutu kontrolü (dosya henüz diske yazılmadan, dosya RAM'deyken!) ---
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell() / (1024 * 1024)  # MB cinsinden
    file.file.seek(0)
    max_file_size_mb = 5
    if file_size > max_file_size_mb: # 5 MB'den büyükse
        errors.append(f"Dosya boyutu çok büyük: {file_size:.2f} MB. Maksimum 5 MB olmalı.")
        try:
            if os.path.exists(file_location):
                os.remove(file_location)
        except Exception as e:
            pass
        return JSONResponse({"status": "error", "errors": errors})
    elif file_size == 0:  # Dosya boşsa
        errors.append("Dosya boş! Lütfen geçerli bir fotoğraf yükleyin.")
        return JSONResponse({"status": "error", "errors": errors})
    
    # --- KURAL 6.1: Dosya boyutu çok küçükse ---
    if file_size < 0.1:  # 100 KB'den küçükse
        errors.append(f"Dosya boyutu çok küçük: {file_size:.2f} MB. Lütfen daha büyük bir fotoğraf yükleyin.") 
        return JSONResponse({"status": "error", "errors": errors})   
    # --- BURADA DOSYAYI UPLOADS KLASÖRÜNE YAZ! ---
    # Pillow ile dosyayı aç ve EXIF verilerini kontrol et
    if img_pil:
        img_pil.save(file_location) # Pillow ile kaydet
    else:
        # Eğer Pillow ile açılamazsa fallback olarak ham dosyayı kaydedebilirsin
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) # Dosyayı uploads klasörüne kaydet

    # --- KURAL 7: FOTOĞRAF BOYUTU ve EN-BOY ORANI KONTROLÜ ---
    img = cv2.imread(file_location)
    if img is None:
        errors.append("Fotoğraf açılamadı!")
        try:
            if os.path.exists(file_location):
                os.remove(file_location)
        except Exception as e:
            pass
        return JSONResponse({"status": "error", "errors": errors})
        
    else:
        h, w = img.shape[:2]
        aspect_ratio = w / h
        if h < 400 or w < 300:
            errors.append(f"Fotoğraf çok küçük ({w}x{h}). Vesikalık için en az 300x400 piksel olmalı.")
        if aspect_ratio < 0.6 or aspect_ratio > 0.9:
            errors.append(f"Fotoğrafın en-boy oranı uygun değil ({aspect_ratio:.2f}). Vesikalık için yaklaşık 3:4 oranı beklenir.")

        # --- KURAL 8: Yüz tespit ve detay kontrolleri ---
        face_count, face_info, num_faces = detect_face(file_location)

        if face_count == -1:
            errors.append("Fotoğraf açılamadı!")
        elif face_count == 0:
            errors.append("Fotoğrafta yüz bulunamadı!")
        elif num_faces > 1:
            errors.append(f"Birden fazla yüz bulundu ({num_faces} tane). Vesikalık için sadece bir yüz olmalı!")

        # --- KURAL 9: Vesikalık özel kriterler (yüz bulunduysa) ---
        if face_info is not None:
            # .1 Yüz çok küçükse
            if face_info["face_ratio"] < 0.2:
                errors.append("Yüz çok küçük. Vesikalık için yakın çekim olmalı.")
            # 8.2 Yüz merkezde değilse
            if face_info["offset_x"] > 0.2 or face_info["offset_y"] > 0.2:
                errors.append("Yüz fotoğrafın ortasında değil.")
            # 8.3 Fotoğraf bulanıksa
            if is_blurry(file_location):
                errors.append("Fotoğraf bulanık veya net değil. Lütfen daha net bir fotoğraf yükleyin.")
            # 8.4 Fotoğraf boydan çekilmiş mi?
            if is_full_body(file_location):
                errors.append("Fotoğraf boydan çekilmiş! Vesikalıkta sadece yüz ve omuzlar görünmeli.")
            # 8.5 Güneş gözlüğü var mı?
            if is_sunglasses_present(file_location):
                errors.append("Fotoğrafta koyu renkli gözlük (güneş gözlüğü) tespit edildi. Vesikalık için gözler açık ve görünür olmalı.")
            # 8.6 Arka plan uygun mu?
            if not is_background_clean(file_location):
                errors.append("Arka plan uygun değil. Vesikalık için açık renkli ve sade arka plan gereklidir.")
            # 8.7 Çok karanlık/parlak mı?
            brightness_status = is_too_dark_or_bright(file_location)
            if brightness_status == "dark":
                errors.append("Fotoğraf çok karanlık. Vesikalık için daha aydınlık bir ortamda çekilmelidir.")
            elif brightness_status == "bright":
                errors.append("Fotoğraf çok parlak. Vesikalık için daha normal ışıkta çekilmelidir.")

        if errors:
            # LOG: Hatalı yükleme kaydı
            logging.info(f"[HATA] {filename} - Hatalar: {errors}")    # Hatalıysa dosyayı otomatik sil (eğer kaydedildiyse)
        
            try: # Eğer error verirse ve dosya kaydedildiyse, sil
                if os.path.exists(file_location): 
                    os.remove(file_location) # Dosyayı sil
            except Exception as e:
                pass  # Sunucu hatası varsa sessizce geç

            return JSONResponse({"status": "error", "errors": errors})
    # LOG: Başarılı yükleme kaydı
    logging.info(
        f"[BAŞARILI] {filename} - Yüz oranı: {face_info['face_ratio'] if face_info else None} | "
        f"Offset: x={face_info['offset_x'] if face_info else None}, y={face_info['offset_y'] if face_info else None}"
    )

    return JSONResponse({
        "status": "ok",
        "file_path": file_location,
        "face_ratio": face_info["face_ratio"] if face_info else None,
        "face_offset_x": face_info["offset_x"] if face_info else None,
        "face_offset_y": face_info["offset_y"] if face_info else None,
        "message": "Fotoğraf vesikalık kurallarına uygun!"
    })

# --- Fotoğrafın bulanık olup olmadığını kontrol eden fonksiyon ---
def is_blurry(image_path, threshold=100):
    """
    Fotoğraf net mi bulanık mı? Laplacian varyans ile ölçülür.
    threshold: Netlik sınırı (daha yüksek = daha net olmalı). 
    Genellikle 100-200 arası iyi çalışır.
    True (bulanık), False (net) döner.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Fotoğrafı gri tonda oku
    if img is None:
        return True  # Dosya açılamazsa bulanık say
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()  # Netlik skorunu hesapla
    return laplacian_var < threshold  # Skor düşükse bulanık kabul et

# --- Fotoğrafta boydan (full body) olup olmadığını anlamak için MediaPipe Pose kullanılır ---
def is_full_body(image_path, threshold=0.6):
    """
    Fotoğrafta boydan vücut (belden aşağı, ör: diz, ayak bileği) görünüyor mu?
    Eğer MediaPipe pose ile diz/ayak/kalça noktaları görünüyorsa, boydan sayılır!
    """
    import mediapipe as mp
    img = cv2.imread(image_path)
    if img is None:
        return False  # Fotoğraf açılamazsa boydan sayma!

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return False  # Vücut hiç algılanmadıysa boydan değildir
        # Kontrol edeceğimiz anahtar noktalar: diz (LEFT_KNEE/RIGHT_KNEE), ayak (LEFT_ANKLE/RIGHT_ANKLE)
        wanted = [
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        h, w, _ = img.shape
        found = 0
        for p in wanted:
            landmark = results.pose_landmarks.landmark[p]
            # 0 ile 1 arasında; eğer x ve y koordinatları %5'ten daha büyükse, kadrajda görünür
            if 0.05 < landmark.x < 0.95 and 0.05 < landmark.y < 0.95:
                found += 1
        # Eğer diz/ayak noktalarının yarısından fazlası kadrajda ise: boydan!
        return found >= 2
def is_sunglasses_present(image_path):
    """
    Fotoğrafta MediaPipe ile göz bölgesi tespit edilir.
    Gözlerin olduğu bölgede koyu ve homojen (simsiyah gibi) bir alan varsa, büyük olasılıkla güneş gözlüğü takılmıştır.
    Basit bir ortalama parlaklık kontrolü yapıyoruz!
    """
    mp_face_mesh = mp.solutions.face_mesh
    img = cv2.imread(image_path)
    if img is None:
        return False  # Fotoğraf açılamazsa hata verme

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            return False  # Yüz algılanmazsa
        h, w, _ = img.shape
        # Göz landmark noktaları (MediaPipe FaceMesh indeksleri)
        left_eye_idx = [33, 133, 159, 145, 153, 154, 155, 133]  # Sol göz çevresi
        right_eye_idx = [362, 263, 387, 373, 380, 381, 382, 362]  # Sağ göz çevresi
        for face_landmarks in result.multi_face_landmarks:
            # Sol ve sağ göz için ayrı ayrı bölge kırpılır
            for eye_idx in [left_eye_idx, right_eye_idx]:
                pts = []
                for idx in eye_idx:
                    pt = face_landmarks.landmark[idx]
                    x, y = int(pt.x * w), int(pt.y * h)
                    pts.append([x, y])
                pts = np.array(pts, np.int32)
                rect = cv2.boundingRect(pts)
                x, y, rw, rh = rect
                eye_roi = img[y:y+rh, x:x+rw]
                if eye_roi.size == 0:
                    continue
                # Göz bölgesinin ortalama parlaklığını bul
                gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray_eye)
                # Eğer bölge aşırı koyuysa (ör. ortalama < 60), büyük olasılıkla koyu gözlük var
                if mean_val < 60:
                    return True
        return False

def is_background_clean(image_path, brightness_threshold=120, color_std_threshold=40):
    """
    Arka plan açık renkli ve sade mi?
    Kenar bölgelerden (üst, alt, sağ, sol) örnek alınır. 
    Ortalama parlaklık düşükse ya da renkler çok değişkense arka plan kirli/koyu sayılır.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False  # Fotoğraf açılamazsa hatalı kabul et
    h, w, _ = img.shape
    border_width = int(min(h, w) * 0.15)  # Kenarlardan %15'lik bir şerit alınır

    # Kenar bölgelerini birleştir
    borders = [
        img[0:border_width, :, :],          # Üst
        img[-border_width:, :, :],          # Alt
        img[:, 0:border_width, :],          # Sol
        img[:, -border_width:, :]           # Sağ
    ]
    border_pixels = np.concatenate([b.reshape(-1, 3) for b in borders], axis=0)

    # Ortalama parlaklık (luminance) ve renk değişkenliği (std) hesapla
    brightness = np.mean(cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY))
    color_std = np.std(border_pixels, axis=0).mean()

    # Eğer kenarlar çok koyuysa ya da çok renkliyse (desenli/karmaşık) arka plan uygun değildir
    if brightness < brightness_threshold or color_std > color_std_threshold:
        return False
    return True

    # --- Fotoğrafın çok karanlık ya da çok parlak olup olmadığını kontrol eden fonksiyon ---
def is_too_dark_or_bright(image_path, dark_threshold=60, bright_threshold=210):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True  # açılamıyorsa, sorun var
    mean = np.mean(img)
    if mean < dark_threshold:
        return "dark"
    elif mean > bright_threshold:
        return "bright"
    return None

# --- Fotoğrafın EXIF verilerini okuyup, oryantasyonu düzeltme ve tarih kontrolü ---
def correct_orientation_and_check_date(file):
    """
    Fotoğrafı PIL ile aç, EXIF'ten tarihi ve oryantasyonu oku.
    - Yan dönmüşse düzeltir, kaydeder.
    - Çok eski (ör: 5 yıldan eski) ise uyarı döndürür.
    """
    errors = []
    # Dosyayı geçici olarak Pillow ile aç
    file.file.seek(0)
    try:
        img = Image.open(file.file)
        # Tarih kontrolü
        exif = img._getexif()
        photo_date = None
        if exif is not None:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'DateTimeOriginal':
                    photo_date = value
                    break
        if photo_date:
            # Tarihi karşılaştır
            photo_datetime = datetime.datetime.strptime(photo_date, '%Y:%m:%d %H:%M:%S')
            now = datetime.datetime.now()
            years_old = (now - photo_datetime).days / 365.25
            if years_old > 5:
                errors.append("Fotoğraf çok eski (5 yıldan önce çekilmiş)! Vesikalık güncel olmalı.")
        # Rotasyon kontrolü
        orientation = None
        if exif is not None:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'Orientation':
                    orientation = value
                    break
        # Rotasyon düzeltmesi (isteğe bağlı)
        if orientation:
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
        return img, errors
    except Exception as e:
        return None, ["EXIF okuma veya döndürme sırasında hata oluştu."]
