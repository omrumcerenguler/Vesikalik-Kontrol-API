from fastapi import FastAPI, File, UploadFile, Request           # FastAPI ana paketleri, dosya upload ve istek için
from fastapi.responses import JSONResponse                      # Kullanıcıya düzgün JSON mesajı döndürmek için
import shutil                                                   # Dosya kopyalamak için
import os                                                      # Dosya/dizin işlemleri için
import cv2                                                     # OpenCV: Resim işleme ve okuma
import mediapipe as mp                                         # MediaPipe: Yüz tespiti için ML tabanlı kütüphane
import numpy as np                                             # NumPy: Matematiksel işlemler ve dizi manipülasyonu
import filetype                                                # Dosya türünü tahmin etmek için
from PIL import Image, ExifTags                                # PIL: Resim işleme ve EXIF verilerini okumak için
import datetime                                                # Tarih ve saat işlemleri için
import uuid                                                    # Benzersiz ID'ler oluşturmak için
import re                                                      # Regex: Dosya adında Türkçe karakter ve özel karakter kontrolü için
from pathlib import Path                                       # Path işlemleri için
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler      # SlowAPI: Rate limiting için
from slowapi.util import get_remote_address                    # SlowAPI: İsteklerin IP adresini almak için  
from slowapi.errors import RateLimitExceeded                   # Rate limit aşımı hatası için
from logging.handlers import RotatingFileHandler               # Log dosyası rotasyonu için

# --- HATA MESAJLARI ---
# Hataları JSON formatında döndürmek için kullanılacak mesajlar
# Her kural için hata mesajları, hem Türkçe hem İngilizce olarak tanımlandı
# Bu mesajlar, kullanıcıya hangi kuralların ihlal edildiğini bildirir.
# Bu mesajlar, hem Türkçe hem İngilizce olarak tanımlandı
# Bu sayede, uygulama çok dilli destek sunabilir.
# Hatalar, kural kodu ile birlikte döndürülür (ör: "KURAL_1").
ERROR_MESSAGES = {
    "KURAL_0": {
    "tr": "Lütfen geçerli bir dil seçiniz. (tr veya en)",
    "en": "Please select a valid language. (tr or en)"
    },
   "KURAL_1": {
        "tr": {
            "turkish": "Dosya adında Türkçe karakter kullanılamaz.",
            "space": "Dosya adında boşluk bulunamaz. Lütfen boşluk yerine '_' veya '-' kullanınız.",
            "length": "Dosya adı çok uzun. Lütfen daha kısa bir dosya adı giriniz.",
            "special": "Dosya adında uygunsuz karakterler var (\\ / : * ? \" < > |). Lütfen bu karakterleri kullanmayın."
        },
        "en": {
            "turkish": "Turkish characters are not allowed in the filename.",
            "space": "Spaces are not allowed in the filename. Please use '_' or '-' instead.",
            "length": "Filename is too long. Please use a shorter name.",
            "special": "Invalid characters in filename (\\ / : * ? \" < > |). Please do not use these."
        }
    },
    "KURAL_2": {
        "tr": "Yüklenen dosya gerçek bir JPEG/PNG görsel değil. Lütfen sadece resim yükleyin.",
        "en": "Uploaded file is not a real JPEG/PNG image. Please upload only image files."
    },
    "KURAL_3": {
        "tr": {
            "ext": "Dosya formatı uygun değil: {ext}. Sadece jpg, jpeg, png yükleyebilirsiniz.",
            "missing": "Dosya adında uzantı yok. Lütfen dosya adını '.jpg' veya '.png' ile bitirin."
        },
        "en": {
            "ext": "Invalid file format: {ext}. Only jpg, jpeg, and png files are allowed.",
            "missing": "No extension in filename. Please end the filename with '.jpg' or '.png'."
        }
    },
    "KURAL_4": {
        "tr": "Dosya uzantısı ile dosyanın gerçek tipi uyuşmuyor.",
        "en": "File extension does not match the real file type."
    },
    "KURAL_5": {
        "tr": {
            "max": "Dosya boyutu çok büyük: {size:.2f} MB. Maksimum 5 MB olmalı.",
            "min": "Dosya boyutu çok küçük: {size:.2f} MB. Lütfen daha büyük bir fotoğraf yükleyin.",
            "empty": "Dosya boş! Lütfen geçerli bir fotoğraf yükleyin."
        },
        "en": {
            "max": "File size too large: {size:.2f} MB. Maximum allowed is 5 MB.",
            "min": "File size too small: {size:.2f} MB. Please upload a larger photo.",
         "empty": "File is empty! Please upload a valid photo."
        }   
    },
    "KURAL_6": {
        "tr": {
            "size": "Fotoğraf çok küçük ({w}x{h}). Vesikalık için en az 300x400 piksel olmalı.",
            "ratio": "Fotoğrafın en-boy oranı uygun değil ({ratio:.2f}). Vesikalık için yaklaşık 3:4 oranı beklenir."
        },
        "en": {
            "size": "Image is too small ({w}x{h}). Minimum required is 300x400 pixels.",
            "ratio": "Invalid aspect ratio ({ratio:.2f}). For ID photo, ratio should be about 3:4."
        }
    },
    "KURAL_8": {
        "tr": {
            "no_face": "Fotoğrafta yüz bulunamadı!",
            "many_faces": "Birden fazla yüz bulundu. Vesikalık için sadece bir yüz olmalı!",
            "small": "Yüz çok küçük. Vesikalık için yakın çekim olmalı.",
            "not_centered": "Yüz fotoğrafın ortasında değil.",
            "cannot_open": "Fotoğraf açılamadı!"
        },
        "en": {
            "no_face": "No face detected in the photo!",
            "many_faces": "More than one face detected. Only one face is allowed for ID photo!",
            "small": "Face is too small. It should be a close-up for ID photo.",
            "not_centered": "Face is not centered in the photo.",
            "cannot_open": "Photo could not be opened!"
        }
    },
    "KURAL_9": {
        "tr": "Fotoğraf bulanık.",
        "en": "Photo is blurry."
    },
    "KURAL_10": {
        "tr": "Fotoğraf boydan çekilmiş. Sadece yüz ve omuzlar görünmeli.",
        "en": "Full-body photo detected. Only face and shoulders should be visible."
    },
    "KURAL_11": {
        "tr": "Fotoğrafta koyu gözlük tespit edildi.",
        "en": "Dark sunglasses detected in the photo."
    },
    " KURAL_12": {
        "tr": {
            "dark": "Arka plan çok koyu. Açık renkli sade arka plan olmalı.",
            "colorful": "Arka plan çok karmaşık veya renkli. Sade ve açık renkli olmalı.",        
        },
        "en": {
            "dark": "Background is too dark. It should be plain and light-colored.",
            "colorful": "Background is too colorful or complex. It should be plain and light-colored.",
        }
    },
    "KURAL_13": {
        "tr": {
            "dark": "Fotoğraf çok karanlık.",
            "bright": "Fotoğraf çok parlak."
        },
        "en": {
            "dark": "Photo is too dark.",
            "bright": "Photo is too bright."
        }
    },
    "KURAL_14": {
        "tr": "Fotoğraf çok eski (5 yıldan önce çekilmiş).",
        "en": "Photo is too old (taken more than 5 years ago)."
    },
}

SUCCESS_MESSAGES = {
    "tr": "Fotoğraf vesikalık kurallarına uygun!",
    "en": "Photo is valid for ID requirements!"
}

# Rate limiter ayarı: IP adresine göre sınır koyar
limiter = Limiter(key_func=get_remote_address)

# --- FastAPI uygulaması başlatma ---
app = FastAPI()                                         # FastAPI uygulamasını başlat
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Log dosyası rotasyonu: 10 MB'a ulaşınca, eskiyi vesikalik_logs.txt.1 olarak saklar, yenisine başlar.
handler = RotatingFileHandler(
    "vesikalik_logs.txt", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']                         # Kabul edilen dosya uzantıları
MIME_MAP = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}
MAX_FILE_SIZE_MB = 5                                                # Maksimum dosya boyutu (MB)
MIN_FILE_SIZE_MB = 0.1                                              # Minimum dosya boyutu (MB)

# --- Sadece test etmek için (zorunlu değil), upload işleviyle ilgisiz ---
@app.post("/kontrol")
async def kontrol(file: UploadFile = File(...)):
    safe_filename = Path(file.filename).name  # Sadece dosya adını al, path silinir
    return {"filename": safe_filename, "message": "Fotoğraf alındı!"}

# --- KURAL 1: Dosya adı ile ilgili kontroller ---
def check_filename_rules(filename, lang):
    errors = []
    # Türkçe karakter kontrolü
    if re.search(r'[çÇğĞıİöÖşŞüÜ]', filename):
        errors.append({"code": "KURAL_1", 
                       "msg": ERROR_MESSAGES["KURAL_1"][lang]["turkish"]})  # Türkçe karakterler
    if " " in filename:
        errors.append({"code": "KURAL_1", 
                       "msg": ERROR_MESSAGES["KURAL_1"][lang]["space"]}) # Boşluk kontrolü
    # Çok uzun dosya adı (ör: >100 karakter)
    if len(filename) > 100:
        errors.append({"code": "KURAL_1", 
                       "msg": ERROR_MESSAGES["KURAL_1"][lang]["length"]})  # Dosya adı çok uzun
    # Uygunsuz özel karakterler
    if re.search(r'[\\/:*?"<>|]', filename):
        errors.append({"code": "KURAL_1", 
                       "msg": ERROR_MESSAGES["KURAL_1"][lang]["special"]})  # Özel karakterler
    return errors

# --- KURAL 2: Dosya tipi ve MIME-Type kontrolü ---
def check_filetype(file, lang):
    file.file.seek(0)
    kind = filetype.guess(file.file.read(261))  # filetype.guess() dosyanın içinden türünü tahmin eder.
    file.file.seek(0)
    errors = []
    if not kind or kind.mime not in ['image/jpeg', 'image/png']:
        errors.append({"code": "KURAL_2", 
                       "msg": ERROR_MESSAGES["KURAL_2"][lang]})
    return errors, kind

# --- KURAL 3: Dosya uzantısı kontrolü ---
def check_extension(filename, lang):
    errors = []
    ext = filename.split('.')[-1].lower()
    # Sadece jpg, jpeg, png kabul et
    if ext not in ALLOWED_EXTENSIONS:
        errors.append({"code": "KURAL_3", 
                       "msg": ERROR_MESSAGES["KURAL_3"][lang]["ext"].format(ext=ext)})
    # Dosya adında uzantı yoksa
    if '.' not in filename:
        errors.append({"code": "KURAL_3", 
                       "msg": ERROR_MESSAGES["KURAL_3"][lang]["missing"]})
    return errors, ext

# --- KURAL 4: Dosya uzantısı ve içeriği uyuşmazlık kontrolü ---
def check_mime_extension(ext, kind, lang):
    errors = []
    if ext in MIME_MAP and kind and kind.mime != MIME_MAP[ext]:
        errors.append({"code": "KURAL_4", 
                       "msg": ERROR_MESSAGES["KURAL_4"][lang]})
    return errors

# --- KURAL 5: Dosya boyutu kontrolü (dosya henüz diske yazılmadan, dosya RAM'deyken!) ---
def check_file_size(file, lang):
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell() / (1024 * 1024)  # MB cinsinden
    file.file.seek(0)
    errors = []
    if file_size > MAX_FILE_SIZE_MB:
        errors.append({"code": "KURAL_5", 
                       "msg": ERROR_MESSAGES["KURAL_5"][lang]["max"].format(size=file_size)})
    elif file_size == 0:
        errors.append({"code": "KURAL_5", 
                       "msg": ERROR_MESSAGES["KURAL_5"][lang]["empty"]}) # Dosya boş
    elif file_size < MIN_FILE_SIZE_MB:
        errors.append({"code": "KURAL_5", 
                       "msg": ERROR_MESSAGES["KURAL_5"][lang]["min"].format(size=file_size)})  # Dosya çok küçük
    return errors, file_size

# --- KURAL 6: Fotoğraf boyutu ve en-boy oranı kontrolü ---
def check_image_dimensions(img, lang):
    errors = []
    h, w = img.shape[:2]                      # Yükseklik (h) ve genişlik (w)
    aspect_ratio = w / h
    
    # Minimum çözünürlük (ör: 300x400)
    if h < 400 or w < 300:
        errors.append({"code": "KURAL_6", 
                       "msg": ERROR_MESSAGES["KURAL_6"][lang]["size"].format(w=w, h=h)})
    # En-boy oranı kontrolü (vesikalıkta genelde 3:4, yani oran 0.75 civarı olmalı)
    if aspect_ratio < 0.6 or aspect_ratio > 0.9:
        errors.append({"code": "KURAL_6", 
                       "msg": ERROR_MESSAGES["KURAL_6"][lang]["ratio"].format(ratio=aspect_ratio)})
    return errors

# --- KURAL 7: Yüz, pozisyon, arka plan, gözlük, parlaklık vb. kontroller ---
def detect_face(image_path, lang):
    mp_face = mp.solutions.face_detection                       # MediaPipe yüz tespit modülünü çağır
    img = cv2.imread(image_path)                                # Dosyayı OpenCV ile oku (resim olarak)
    if img is None:
        return -1, None, 0                                     # Resim açılamadıysa: hata kodu dön
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # MediaPipe için resmi RGB'ye çevir
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
            face_center_x = bboxC.xmin * w + box_w / 2          # Yüzün merkezi nerde? (x,y)
            face_center_y = bboxC.ymin * h + box_h / 2
            center_x, center_y = w / 2, h / 2                   # Fotoğrafın merkezi nerede?
            offset_x = abs(face_center_x - center_x) / w        # Yüzün merkezi ile fotoğraf merkezi arasındaki fark (0 ise tam ortada)
            offset_y = abs(face_center_y - center_y) / h
            return 1, {"face_ratio": face_ratio, "offset_x": offset_x, "offset_y": offset_y}, num_faces
        else:
            return 0, None, 0
# --- KURAL 8: Yüz tespiti ve vesikalık kurallarını kontrol eden fonksiyon ---
def check_face_rules(image_path, lang):
    errors = []
    face_count, face_info, num_faces = detect_face(image_path, lang)  # Yüz tespit et
    if face_count == -1:
        errors.append({"code": "KURAL_8", 
                       "msg": ERROR_MESSAGES["KURAL_8"][lang]["cannot_open"]})  # Fotoğraf açılamadı!
    elif face_count == 0:
        errors.append({"code": "KURAL_8", 
                       "msg":ERROR_MESSAGES["KURAL_8"][lang]["no_face"]})  # Yüz bulunamadı!
    elif num_faces > 1:
        errors.append({"code": "KURAL_8", 
                       "msg": ERROR_MESSAGES["KURAL_8"][lang]["many_faces"]})  # Birden fazla yüz bulundu!
    if face_info is not None:
        # Yüz çok küçükse (vesikalıkta yüz büyük olmalı!)
        if face_info["face_ratio"] < 0.2:
            errors.append({"code": "KURAL_8", 
                           "msg": ERROR_MESSAGES["KURAL_8"][lang]["small"]})  # Yüz çok küçük!
        # Yüz merkezde değilse (vesikalıkta yüz neredeyse tam ortada olmalı)
        if face_info["offset_x"] > 0.2 or face_info["offset_y"] > 0.2:
            errors.append({"code": "KURAL_8", 
                           "msg": ERROR_MESSAGES["KURAL_8"][lang]["not_centered"]})  # Yüz fotoğrafın ortasında değil!
    return errors, face_info

# --- KURAL 9: Fotoğrafın bulanık olup olmadığını kontrol eden fonksiyon ---
def is_blurry(image_path, threshold=100):
    """
    Fotoğraf net mi bulanık mı? Laplacian varyans ile ölçülür.
    threshold: Netlik sınırı (daha yüksek = daha net olmalı). 
    Genellikle 100-200 arası iyi çalışır.
    True (bulanık), False (net) döner.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Fotoğrafı gri tonda oku
    if img is None:
        return True  # Dosya açılamazsa bulanık say. True bulanık olduğunu gösterir bu fonksiyon için.  
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()  # Netlik skorunu hesapla
    return laplacian_var < threshold  # Skor düşükse bulanık kabul et

# --- KURAL 10: Fotoğrafta boydan (full body) olup olmadığını anlamak için MediaPipe Pose kullanılır ---
def is_full_body(image_path, threshold=0.6):
    """
    Fotoğrafta boydan vücut (belden aşağı, ör: diz, ayak bileği) görünüyor mu?
    Eğer MediaPipe pose ile diz/ayak/kalça noktaları görünüyorsa, boydan sayılır!
    """
    import mediapipe as mp
    img = cv2.imread(image_path)
    if img is None:
        return False # Fotoğraf açılamazsa boydan sayılmaz
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if not results.pose_landmarks: 
            return False 
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
            if 0.05 < landmark.x < 0.95 and 0.05 < landmark.y < 0.95:
                found += 1
        return found >= 2
# --- KURAL 11: Fotoğrafta gözlük (özellikle güneş gözlüğü) var mı? ---
def is_sunglasses_present(image_path):
    """
    Fotoğrafta MediaPipe ile göz bölgesi tespit edilir.
    Gözlerin olduğu bölgede koyu ve homojen (simsiyah gibi) bir alan varsa, büyük olasılıkla güneş gözlüğü takılmıştır.
    Basit bir ortalama parlaklık kontrolü yapıyoruz!
    """
    mp_face_mesh = mp.solutions.face_mesh
    img = cv2.imread(image_path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(img_rgb)
        if not result.multi_face_landmarks:
            return False
        h, w, _ = img.shape
        left_eye_idx = [33, 133, 159, 145, 153, 154, 155, 133]
        right_eye_idx = [362, 263, 387, 373, 380, 381, 382, 362]
        for face_landmarks in result.multi_face_landmarks:
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
                gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray_eye)
                if mean_val < 60:
                    return True
        return False
# --- KURAL 12: Arka planın temiz ve sade olup olmadığını kontrol et ---
def is_background_clean(image_path, brightness_threshold=120, color_std_threshold=40):
    """
    Arka plan açık renkli ve sade mi?
    Kenar bölgelerden (üst, alt, sağ, sol) örnek alınır. 
    Ortalama parlaklık düşükse ya da renkler çok değişkense arka plan kirli/koyu sayılır.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "error"
    h, w, _ = img.shape
    border_width = int(min(h, w) * 0.15)
    borders = [
        img[0:border_width, :, :],
        img[-border_width:, :, :],
        img[:, 0:border_width, :],
        img[:, -border_width:, :]
    ]
    border_pixels = np.concatenate([b.reshape(-1, 3) for b in borders], axis=0)
    brightness = np.mean(cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY))
    color_std = np.std(border_pixels, axis=0).mean()
    if brightness < brightness_threshold:
        return "dark"
    if color_std > color_std_threshold:
        return "colorful"
    return "clean"
# --- KURAL 13: Fotoğrafın çok karanlık veya çok parlak olup olmadığını kontrol et ---
def is_too_dark_or_bright(image_path, dark_threshold=60, bright_threshold=210):
    """
    Fotoğraf çok karanlık veya çok parlak mı? Ortalama parlaklık ile ölçülür.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    mean = np.mean(img)
    if mean < dark_threshold:
        return "dark"
    elif mean > bright_threshold:
        return "bright"
    return None

# --- KURAL 14: Fotoğrafın EXIF verilerini okuyup, oryantasyonu düzeltme ve tarih kontrolü ---
def correct_orientation_and_check_date(file, lang):
    """
    Fotoğrafı PIL ile aç, EXIF'ten tarihi ve oryantasyonu oku.
    - Yan dönmüşse düzeltir, kaydeder.
    - Çok eski (ör: 5 yıldan eski) ise uyarı döndürür.
    """
    errors = []
    file.file.seek(0)
    try:
        img = Image.open(file.file)
        exif = img._getexif()
        photo_date = None
        if exif is not None:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'DateTimeOriginal':
                    photo_date = value
                    break
        if photo_date:
            photo_datetime = datetime.datetime.strptime(photo_date, '%Y:%m:%d %H:%M:%S')
            now = datetime.datetime.now()
            years_old = (now - photo_datetime).days / 365.25
            if years_old > 5:
                errors.append({"code": "KURAL_14", 
                               "msg": ERROR_MESSAGES["KURAL_14"][lang]})
        orientation = None
        if exif is not None:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'Orientation':
                    orientation = value
                    break
        if orientation:
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
        return img, errors
    except Exception as e:
        return None, [{"code": "KURAL_14", "msg": ERROR_MESSAGES["KURAL_14"][lang]}]
# --- ANA ENDPOINT: Fotoğrafı upload et, tüm vesikalık kontrollerini sırayla yap ---
@app.post("/upload/")
@limiter.limit("30/minute")   # ← Her IP dakikada 30 yükleme hakkı
async def upload_image(request: Request, file: UploadFile = File(...)):
    errors = []

    lang = request.query_params.get("lang", "tr")
    if lang not in ["tr", "en"]:
        return JSONResponse({
            "status": "error",
            "errors": [{"code": "KURAL_0",
                        "msg": "Lütfen geçerli bir dil seçiniz. (tr veya en)"
            }]
        })

    # Benzersiz ve güvenli dosya adı oluştur
    original_filename = Path(file.filename).name   # Sadece dosya adını al, path'i sil
    unique_id = uuid.uuid4().hex[:8]              # 8 karakterlik random id
    filename = f"{unique_id}_{original_filename}" # Yeni güvenli ve benzersiz dosya adın

    # --- KURAL 1: Dosya adı kontrolleri ---
    errors.extend(check_filename_rules(filename, lang))

    # --- KURAL 2: Dosya tipi ve MIME-Type kontrolü ---
    err, kind = check_filetype(file, lang) 
    errors.extend(err)

    # --- KURAL 3: Dosya uzantısı kontrolü ---
    err, ext = check_extension(filename, lang)
    errors.extend(err)

    # --- KURAL 4: Uzantı-mime uyumu (Dosya uzantısı ve içeriği uyuşmazlık kontrolü) ---
    errors.extend(check_mime_extension(ext, kind, lang))

    # --- KURAL 5: Dosya boyutu kontrolü (henüz RAM'deyken, diske yazılmadan) ---
    err, file_size = check_file_size(file, lang)
    errors.extend(err)

    # --- KURAL 14: EXIF meta verisi ve rotasyon kontrolü ---
    img_pil, exif_errors = correct_orientation_and_check_date(file, lang)
    if exif_errors:
        errors.extend(exif_errors)

    # uploads klasörü yoksa oluştur
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_location = f"uploads/{filename}"

    # KRİTİK HATALAR ---
    # Eğer kritik hatalar varsa, dosyayı kaydetme, logla ve hata dön
    # Kritik hatalar: fotoğraf açılamadı, boş dosya, yanlış tür
    kritik_hatalar = [
    ERROR_MESSAGES["KURAL_8"][lang]["cannot_open"], # Fotoğraf açılamadı!
    ERROR_MESSAGES["KURAL_5"][lang]["empty"], # Dosya boş!
    ERROR_MESSAGES["KURAL_2"][lang] # Yüklenen dosya gerçek bir JPEG/PNG görsel değil.
    ]
    # Substring ile ekstra kontrol
    # Eğer hata mesajları arasında kritik hatalardan biri varsa veya dosya uzantısı ile gerçek tip uyuşmuyorsa
    # (ör: dosya adı .jpg ama içeriği png ise), hata dön
    if (
        any(isinstance(h, dict) and h.get("msg") in kritik_hatalar for h in errors) 
        or any(
            isinstance(h, dict) and "uzantısı ile dosyanın gerçek tipi" in h.get("msg", "")
            for h in errors
        )
    ):
        logging.info(f"[HATA-KRİTİK] {filename} - Hatalar: {errors}")
        return JSONResponse({"status": "error", "errors": errors}) 
    
    # Hata varsa kaydetme, logla ve dön
    if errors:
        logging.info(f"[HATA] {filename} - Hatalar: {errors}")
        return JSONResponse({"status": "error", "errors": errors})

    # Dosyayı kaydet (Pillow ile ya da ham şekilde)
    if img_pil:
        img_pil.save(file_location)
    else:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # --- KURAL 6: Fotoğraf boyutu ve en-boy oranı ---
    img = cv2.imread(file_location)
    if img is None:
        errors.append({"code": "KURAL_6", 
                       "msg": ERROR_MESSAGES["KURAL_6"][lang]["size"].format(w="?", h="?")})  # Fotoğraf okunamadı!
    else:
        errors.extend(check_image_dimensions(img, lang))

    # --- KURAL 8: Yüz tespiti ve vesikalık kuralları kontrolleri  ---
    err, face_info = check_face_rules(file_location, lang)
    errors.extend(err)

    # --- KURAL 9-13: Diğer özel kurallar (fotoğraf bulanıklığı, boydan, gözlük, arka plan, parlaklık) ---
    if face_info is not None: 
        if is_blurry(file_location): 
            errors.append({"code": "KURAL_9", 
                           "msg": ERROR_MESSAGES["KURAL_9"][lang]})  # Fotoğraf bulanık!
        if is_full_body(file_location):
            errors.append({"code": "KURAL_10", 
                           "msg": ERROR_MESSAGES["KURAL_10"][lang]}) # Fotoğraf boydan çekilmiş!
        if is_sunglasses_present(file_location):
            errors.append({"code": "KURAL_11", 
                           "msg": ERROR_MESSAGES["KURAL_11"][lang]}) # Fotoğrafta koyu gözlük tespit edildi!
        bg_status = is_background_clean(file_location)
        if bg_status == "dark":
            errors.append({"code": "KURAL_12", 
                           "msg": ERROR_MESSAGES["KURAL_12"][lang]["dark"]})
        elif bg_status == "colorful":
            errors.append({"code": "KURAL_12", 
                           "msg": ERROR_MESSAGES["KURAL_12"][lang]["colorful"]}) # "clean" ise hata ekleme!
        brightness_status = is_too_dark_or_bright(file_location)
        if brightness_status == "dark":
            errors.append({"code": "KURAL_13", 
                           "msg": ERROR_MESSAGES["KURAL_13"][lang]["dark"]}) # Fotoğraf çok karanlık!
        elif brightness_status == "bright":
            errors.append({"code": "KURAL_13", 
                           "msg": ERROR_MESSAGES["KURAL_13"][lang]["bright"]}) # Fotoğraf çok parlak!

    # Hatalıysa dosyayı sil, logla ve hata dön
    if errors:
        logging.info(f"[HATA] {filename} - Hatalar: {errors}")
        try:
            if os.path.exists(file_location):
                os.remove(file_location)
        except Exception:
            pass
        return JSONResponse({"status": "error", "errors": errors})

    # Başarılı log ve dönüş
    logging.info(
        f"[BAŞARILI] {filename} - Yüz oranı: {face_info['face_ratio'] if face_info else None} | "
        f"Offset: x={face_info['offset_x'] if face_info else None}, y={face_info['offset_y'] if face_info else None}"
    )

    return JSONResponse({
        "status": "ok",
        "file_path": file_location,                  # Sunucudaki dosya yolu
        "face_ratio": face_info["face_ratio"] if face_info else None,       # Yüz oranı (büyüklüğü)
        "face_offset_x": face_info["offset_x"] if face_info else None,      # Yüzün yatay merkeze yakınlığı
        "face_offset_y": face_info["offset_y"] if face_info else None,      # Yüzün dikey merkeze yakınlığı
        "message": SUCCESS_MESSAGES[lang] 
    })