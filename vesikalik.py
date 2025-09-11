from starlette.responses import Response
try:
    # Pylance sometimes cannot resolve attributes on mediapipe.solutions.
    # Import an explicit alias and type it as Any to keep type-checkers calm.
    from mediapipe import solutions as mp_solutions  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - import is safe at runtime
    mp_solutions = None  # type: ignore[assignment]
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import JSONResponse
 # Wrapper to satisfy FastAPI's expected exception handler signature
def rate_limit_handler(request: Request, exc: Exception) -> Response:
    # slowapi exposes a concrete handler that expects RateLimitExceeded
    # FastAPI wants a handler typed as (Request, Exception) -> Response
    # We assert at runtime and delegate to slowapi's default implementation.
    assert isinstance(exc, RateLimitExceeded)
    return _rate_limit_exceeded_handler(request, exc)  # type: ignore[arg-type]
import cv2
import numpy as np
from PIL import Image, ExifTags
import filetype
from pathlib import Path
import datetime
import logging
from logging.handlers import RotatingFileHandler
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
    "KURAL_12": {
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
    "SYSTEM": {
        "tr": "Bilinmeyen bir sistem hatası oluştu. Lütfen tekrar deneyin.",
        "en": "An unknown system error occurred. Please try again."
    },
}

SUCCESS_MESSAGES = {
    "tr": "Fotoğraf vesikalık kurallarına uygun!",
    "en": "Photo is valid for ID requirements!"
}

# Rate limiter ayarı: IP adresine göre sınır koyar
limiter = Limiter(key_func=get_remote_address)

# --- FastAPI uygulaması başlatma ---
# FastAPI uygulamasını başlat
app = FastAPI()
app.state.limiter = limiter
from fastapi import Request
from fastapi.responses import JSONResponse

# Use the default slowapi rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# Log dosyası rotasyonu: 10 MB'a ulaşınca, eskiyi vesikalik_logs.txt.1 olarak saklar, yenisine başlar.
handler = RotatingFileHandler(
    "vesikalik_logs.txt", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Kabul edilen dosya uzantıları
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
MIME_MAP = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}
# Maksimum dosya boyutu (MB)
MAX_FILE_SIZE_MB = 5
# Minimum dosya boyutu (MB)
MIN_FILE_SIZE_MB = 0.1

# --- Sadece test etmek için (zorunlu değil), upload işleviyle ilgisiz ---


@app.post("/kontrol")
async def kontrol(file: UploadFile = File(...)):
    # Sadece dosya adını al, path silinir
    safe_filename = Path(file.filename if file.filename is not None else "unknown.jpg").name
    return JSONResponse({"status": "ok", "filename": safe_filename, "message": "Fotoğraf alındı!"})

# --- KURAL 1: Dosya adı kontrolleri ---
def check_filename_rules(filename, lang):
    errors = []
    # Türkçe karakter kontrolü
    import re
    if re.search(r'[çÇğĞıİöÖşŞüÜ]', filename):
        errors.append({"code": "KURAL_1", "msg": ERROR_MESSAGES["KURAL_1"][lang]["turkish"]})
    # Boşluk kontrolü
    if " " in filename:
        errors.append({"code": "KURAL_1", "msg": ERROR_MESSAGES["KURAL_1"][lang]["space"]})
    # Dosya adı uzunluk kontrolü
    if len(filename) > 100:
        errors.append({"code": "KURAL_1", "msg": ERROR_MESSAGES["KURAL_1"][lang]["length"]})
    # Özel karakter kontrolü
    if re.search(r'[\\/:*?"<>|]', filename):
        errors.append({"code": "KURAL_1", "msg": ERROR_MESSAGES["KURAL_1"][lang]["special"]})
    return errors

# --- KURAL 2: Dosya tipi ve MIME-Type kontrolü ---
def check_filetype(file, lang):
    file.file.seek(0)
    kind = filetype.guess(file.file.read(261))  # Dosya tipi kontrolü
    file.file.seek(0)
    errors = []
    if not kind or kind.mime not in ['image/jpeg', 'image/png']:
        errors.append({"code": "KURAL_2", "msg": ERROR_MESSAGES["KURAL_2"][lang]})
    return errors, kind

# --- KURAL 3: Dosya uzantısı kontrolü ---
def check_extension(filename, lang):
    errors = []
    ext = filename.split('.')[-1].lower()
    # Dosya uzantısı kontrolü
    if ext not in ALLOWED_EXTENSIONS:
        errors.append({"code": "KURAL_3", "msg": ERROR_MESSAGES["KURAL_3"][lang]["ext"].format(ext=ext)})
    if '.' not in filename:
        errors.append({"code": "KURAL_3", "msg": ERROR_MESSAGES["KURAL_3"][lang]["missing"]})
    return errors, ext

# --- KURAL 4: Dosya uzantısı ve içeriği uyuşmazlık kontrolü ---
def check_mime_extension(ext, kind, lang):
    errors = []
    # Uzantı-mime uyumu kontrolü
    if ext in MIME_MAP and kind and kind.mime != MIME_MAP[ext]:
        errors.append({"code": "KURAL_4", "msg": ERROR_MESSAGES["KURAL_4"][lang]})
    return errors

# --- KURAL 5: Dosya boyutu kontrolü ---
def check_file_size(file, lang):
    import os
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell() / (1024 * 1024)  # MB cinsinden
    file.file.seek(0)
    errors = []
    if file_size > MAX_FILE_SIZE_MB:
        errors.append({"code": "KURAL_5", "msg": ERROR_MESSAGES["KURAL_5"][lang]["max"].format(size=file_size)})
    elif file_size == 0:
        errors.append({"code": "KURAL_5", "msg": ERROR_MESSAGES["KURAL_5"][lang]["empty"]})
    return errors, file_size

# --- KURAL 6: Fotoğraf boyutu ve en-boy oranı kontrolü ---
def check_image_dimensions(img, lang):
    errors = []
    h, w = img.shape[:2]  # Yükseklik ve genişlik
    aspect_ratio = w / h
    # Minimum çözünürlük kontrolü
    if h < 400 or w < 300:
        errors.append({"code": "KURAL_6", "msg": ERROR_MESSAGES["KURAL_6"][lang]["size"].format(w=w, h=h)})
    # En-boy oranı kontrolü
    if aspect_ratio < 0.6 or aspect_ratio > 0.9:
        errors.append({"code": "KURAL_6", "msg": ERROR_MESSAGES["KURAL_6"][lang]["ratio"].format(ratio=aspect_ratio)})
    return errors

# --- KURAL 7: Yüz tespit fonksiyonu ---
def detect_face(image_path, lang):
    img = cv2.imread(image_path)
    if img is None:
        return -1, None, 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:  # type: ignore[attr-defined]
        result = face_detection.process(img_rgb)
        if result.detections:
            num_faces = len(result.detections)
            detection = result.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            box_w, box_h = bboxC.width * w, bboxC.height * h
            face_area = box_w * box_h
            img_area = w * h
            face_ratio = face_area / img_area
            face_center_x = bboxC.xmin * w + box_w / 2
            face_center_y = bboxC.ymin * h + box_h / 2
            center_x, center_y = w / 2, h / 2
            offset_x = abs(face_center_x - center_x) / w
            offset_y = abs(face_center_y - center_y) / h
            return 1, {"face_ratio": face_ratio, "offset_x": offset_x, "offset_y": offset_y}, num_faces
        else:
            return 0, None, 0
# --- KURAL 8: Yüz tespiti ve vesikalık kuralları kontrolü ---
def check_face_rules(image_path, lang):
    errors = []
    face_count, face_info, num_faces = detect_face(image_path, lang)
    if face_count == -1:
        errors.append({"code": "KURAL_8", "msg": ERROR_MESSAGES["KURAL_8"][lang]["cannot_open"]})
    elif face_count == 0:
        errors.append({"code": "KURAL_8", "msg": ERROR_MESSAGES["KURAL_8"][lang]["no_face"]})
    elif num_faces > 1:
        errors.append({"code": "KURAL_8", "msg": ERROR_MESSAGES["KURAL_8"][lang]["many_faces"]})
    if face_info is not None:
        # Yüz boyutu kontrolü
        if face_info["face_ratio"] < 0.12:
            errors.append({"code": "KURAL_8", "msg": ERROR_MESSAGES["KURAL_8"][lang]["small"]})
        # Yüz merkezde mi kontrolü
        if face_info["offset_x"] > 0.3 or face_info["offset_y"] > 0.3:
            errors.append({"code": "KURAL_8", "msg": ERROR_MESSAGES["KURAL_8"][lang]["not_centered"]})
    return errors, face_info

# --- KURAL 9: Fotoğraf bulanıklık kontrolü ---
def is_blurry(image_path, threshold=250):
    # Fotoğraf bulanıklık kontrolü
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold

# --- KURAL 10: Boydan fotoğraf kontrolü ---
def is_full_body(image_path, threshold=0.6):
    img = cv2.imread(image_path)
    if img is None:
        return False
    mp_pose = mp_solutions.pose  # type: ignore[attr-defined]
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
        found = 0
        for p in wanted:
            landmark = results.pose_landmarks.landmark[p]
            if 0.05 < landmark.x < 0.95 and 0.05 < landmark.y < 0.95:
                found += 1
        return found >= 2
# --- KURAL 11: Gözlük kontrolü ---
def is_sunglasses_present(image_path):
    face_mesh_module = mp_solutions.face_mesh  # type: ignore[attr-defined]
    img = cv2.imread(image_path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with face_mesh_module.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
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
# --- KURAL 12: Arka plan kontrolü ---
def is_background_clean(image_path, brightness_threshold=80, color_std_threshold=100):
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
# --- KURAL 13: Parlaklık/karanlık kontrolü ---
def is_too_dark_or_bright(image_path, dark_threshold=50, bright_threshold=225):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    mean = np.mean(img)
    if mean < dark_threshold:
        return "dark"
    elif mean > bright_threshold:
        return "bright"
    return None

# --- KURAL 14: EXIF ve tarih kontrolü ---
def correct_orientation_and_check_date(file, lang):
    errors = []
    file.file.seek(0)
    try:
        img = Image.open(file.file)
        exif = img.getexif()
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
            if years_old > 10:
                errors.append({"code": "KURAL_14", "msg": ERROR_MESSAGES["KURAL_14"][lang]})
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
    except Exception:
        return None, [{"code": "KURAL_14", "msg": ERROR_MESSAGES["KURAL_14"][lang]}]
# --- ANA ENDPOINT: Fotoğrafı yükle ve vesikalık kurallarını sırayla kontrol et ---
@app.post("/upload/")
@limiter.limit("30/minute")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    lang: str = Query("tr", description="Dil seçimi: 'tr' veya 'en'")
):
    import os
    import uuid
    import shutil
    try:
        errors = []
        sebep = ""

        # Dil kontrolü
        if lang not in ["tr", "en"]:
            sebep = "Lütfen geçerli bir dil seçiniz. (tr veya en)"
            return JSONResponse({
                "status": "error",
                "errors": [{"code": "KURAL_0", "msg": sebep}]
            })

        # Dosya adı oluşturma
        original_filename = Path(file.filename or "unknown.jpg").name
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{unique_id}_{original_filename}"

        # KURAL 1: Dosya adı kontrolleri
        err_filename_rules = check_filename_rules(filename, lang)
        for err in err_filename_rules:
            if not sebep:
                sebep = err.get("msg", "")
            errors.append({"code": err.get("code", "KURAL_1"), "msg": err.get("msg", "")})

        # KURAL 2: Dosya tipi kontrolü
        err, kind = check_filetype(file, lang)
        for e in err:
            if not sebep:
                sebep = e.get("msg", "")
            errors.append({"code": e.get("code", "KURAL_2"), "msg": e.get("msg", "")})

        # KURAL 3: Dosya uzantısı kontrolü
        err, ext = check_extension(filename, lang)
        for e in err:
            if not sebep:
                sebep = e.get("msg", "")
            errors.append({"code": e.get("code", "KURAL_3"), "msg": e.get("msg", "")})

        # KURAL 4: Uzantı-mime uyumu kontrolü
        err_mime = check_mime_extension(ext, kind, lang)
        for e in err_mime:
            if not sebep:
                sebep = e.get("msg", "")
            errors.append({"code": e.get("code", "KURAL_4"), "msg": e.get("msg", "")})

        # KURAL 5: Dosya boyutu kontrolü
        err, file_size = check_file_size(file, lang)
        for e in err:
            if not sebep:
                sebep = e.get("msg", "")
            errors.append({"code": e.get("code", "KURAL_5"), "msg": e.get("msg", "")})

        # KURAL 14: EXIF ve tarih kontrolü
        img_pil, exif_errors = correct_orientation_and_check_date(file, lang)
        if exif_errors:
            for e in exif_errors:
                if not sebep:
                    sebep = e.get("msg", "")
                errors.append({"code": e.get("code", "KURAL_14"), "msg": e.get("msg", "")})

        # uploads klasörü yoksa oluştur
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        file_location = f"uploads/{filename}"

        # Kritik hatalar: fotoğraf açılamadı, boş dosya, yanlış tür
        kritik_hatalar = [
            ERROR_MESSAGES["KURAL_8"][lang]["cannot_open"],
            ERROR_MESSAGES["KURAL_5"][lang]["empty"],
            ERROR_MESSAGES["KURAL_2"][lang]
        ]
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

        # Dosyayı kaydet
        if img_pil:
            img_pil.save(file_location)
        else:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # KURAL 6: Fotoğraf boyutu ve en-boy oranı
        img = cv2.imread(file_location)
        if img is None:
            sebep = ERROR_MESSAGES["KURAL_6"][lang]["size"].format(w="?", h="?")
            errors.append({"code": "KURAL_6", "msg": sebep})
        else:
            err_dim = check_image_dimensions(img, lang)
            for e in err_dim:
                if not sebep:
                    sebep = e.get("msg", "")
                errors.append({"code": e.get("code", "KURAL_6"), "msg": e.get("msg", "")})

        # KURAL 8: Yüz tespiti ve kurallar
        err, face_info = check_face_rules(file_location, lang)
        for e in err:
            if not sebep:
                sebep = e.get("msg", "")
            errors.append({"code": e.get("code", "KURAL_8"), "msg": e.get("msg", "")})

        # KURAL 9-13: Diğer özel kurallar
        if face_info is not None:
            if is_blurry(file_location):
                sebep = ERROR_MESSAGES["KURAL_9"][lang]
                errors.append({"code": "KURAL_9", "msg": sebep})
            if is_full_body(file_location):
                sebep = ERROR_MESSAGES["KURAL_10"][lang]
                errors.append({"code": "KURAL_10", "msg": sebep})
            if is_sunglasses_present(file_location):
                sebep = ERROR_MESSAGES["KURAL_11"][lang]
                errors.append({"code": "KURAL_11", "msg": sebep})
            bg_status = is_background_clean(file_location)
            if bg_status == "dark":
                sebep = ERROR_MESSAGES["KURAL_12"][lang]["dark"]
                errors.append({"code": "KURAL_12", "msg": sebep})
            elif bg_status == "colorful":
                sebep = ERROR_MESSAGES["KURAL_12"][lang]["colorful"]
                errors.append({"code": "KURAL_12", "msg": sebep})
            brightness_status = is_too_dark_or_bright(file_location)
            if brightness_status == "dark":
                sebep = ERROR_MESSAGES["KURAL_13"][lang]["dark"]
                errors.append({"code": "KURAL_13", "msg": sebep})
            elif brightness_status == "bright":
                sebep = ERROR_MESSAGES["KURAL_13"][lang]["bright"]
                errors.append({"code": "KURAL_13", "msg": sebep})

        # Hatalıysa dosyayı sil, logla ve hata dön
        if errors:
            logging.info(f"[HATA] {filename} - Hatalar: {errors}")
            try:
                if os.path.exists(file_location):
                    os.remove(file_location)
            except Exception:
                pass
            return JSONResponse({
                "status": "error",
                "errors": errors
            })

        # Başarılı log ve dönüş
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
            "message": SUCCESS_MESSAGES[lang],
            "original_filename": original_filename,
            "size_mb": round(file_size, 2),
            "upload_time": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        user_msg = ERROR_MESSAGES["SYSTEM"][lang]
        logging.error(
            f"[SİSTEM HATASI] {file.filename if file else '-'} - {e}",
            exc_info=True
        )
        return JSONResponse({
            "status": "error",
            "errors": [{"code": "SYSTEM", "msg": user_msg}]
        })

API_VERSION = "v1.1.1"
BUILD_DATE = "2025-07-24"


@app.get("/version")
async def version():
    return {
        "version": API_VERSION,
        "build_date": BUILD_DATE
    }
