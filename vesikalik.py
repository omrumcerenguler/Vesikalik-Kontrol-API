from fastapi import FastAPI, File, UploadFile            # FastAPI ana paketleri, dosya upload için
from fastapi.responses import JSONResponse               # Kullanıcıya düzgün JSON mesajı döndürmek için
import shutil                                            # Dosya kopyalamak için
import os                                               # Dosya/dizin işlemleri için
import cv2                                              # OpenCV: Resim işleme ve okuma
import mediapipe as mp                                  # MediaPipe: Yüz tespiti için ML tabanlı kütüphane
import numpy as np 

app = FastAPI()                                         # FastAPI uygulamasını başlat

# --- Sadece test etmek için (zorunlu değil), upload işleviyle ilgisiz ---
@app.post("/kontrol")
async def kontrol(file: UploadFile = File(...)):
    # Kullanıcıdan dosya geliyor mu kontrolü için basit cevap
    return {"filename": file.filename, "message": "Fotoğraf alındı!"}

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
async def upload_image(file: UploadFile = File(...)):
    # 1. uploads klasörü yoksa oluştur (ilk upload'da çalışır)
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_location = f"uploads/{file.filename}"                  # Dosyanın nereye kaydedileceği

    # 2. Dosyayı kullanıcının upload ettiği şekilde sunucuya kaydet
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
        # --- FOTOĞRAF BOYUTU ve EN-BOY ORANI KONTROLÜ ---
    img = cv2.imread(file_location)  # Fotoğrafı oku
    if img is None:
        return JSONResponse({"status": "error", "detail": "Fotoğraf açılamadı!"})

    h, w = img.shape[:2]  # Yükseklik (h) ve genişlik (w)
    
    # Minimum çözünürlük (ör: 300x400)
    if h < 400 or w < 300:
        return JSONResponse({
            "status": "error",
            "detail": f"Fotoğraf çok küçük ({w}x{h}). Vesikalık için en az 300x400 piksel olmalı."
        })

    # En-boy oranı kontrolü (vesikalıkta genelde 3:4, yani oran 0.75 civarı olmalı)
    aspect_ratio = w / h
    if aspect_ratio < 0.6 or aspect_ratio > 0.9:
        return JSONResponse({
            "status": "error",
            "detail": f"Fotoğrafın en-boy oranı uygun değil ({aspect_ratio:.2f}). Vesikalık için yaklaşık 3:4 oranı beklenir."
        })
    

    # 3. Yüz tespit et ve detayları al
    face_count, face_info, num_faces = detect_face(file_location)

    # --- Hatalar ve özel durumlar için erken dönüşler ---
    if face_count == -1:
        # Dosya bozuk ya da resim değil
        return JSONResponse({"status": "error", "detail": "Fotoğraf açılamadı!"})
    elif face_count == 0:
        # Hiç yüz yok
        return JSONResponse({"status": "error", "detail": "Fotoğrafta yüz bulunamadı!"})
    elif num_faces > 1:
        # Birden fazla yüz varsa, vesikalık olamaz!
        return JSONResponse({
            "status": "error",
            "detail": f"Birden fazla yüz bulundu ({num_faces} tane). Vesikalık için sadece bir yüz olmalı!"
        })

    # --- Vesikalık kriterleri ---
    # Kural 1: Yüz çok küçükse (vesikalıkta yüz büyük olmalı!)
    if face_info["face_ratio"] < 0.2:
        return JSONResponse({
            "status": "error",
            "detail": "Yüz çok küçük. Vesikalık için yakın çekim olmalı."
        })
    # Kural 2: Yüz merkezde değilse (vesikalıkta yüz neredeyse tam ortada olmalı)
    if face_info["offset_x"] > 0.2 or face_info["offset_y"] > 0.2:
        return JSONResponse({
            "status": "error",
            "detail": "Yüz fotoğrafın ortasında değil."
        })
    # Kural 3: Fotoğraf bulanıksa (net değilse) vesikalık olamaz!
    if is_blurry(file_location):
        return JSONResponse({
            "status": "error",
            "detail": "Fotoğraf bulanık veya net değil. Lütfen daha net bir fotoğraf yükleyin."
        })
    # Kural 4: Fotoğraf boydan çekilmiş mi? (vesikalıkta yalnızca yüz ve omuzlar görünmeli!)
    # MediaPipe Pose kullanarak diz/ayak bileği noktaları kadrajda mı kontrol edilir.
    # Eğer bu noktaların en az yarısı kadrajda görünüyorsa, fotoğraf boydan sayılır ve hata döndürülür.
    if is_full_body(file_location):
        return JSONResponse({
            "status": "error",
            "detail": "Fotoğraf boydan çekilmiş! Vesikalıkta sadece yüz ve omuzlar görünmeli."
        })
    # Kural 5: Fotoğrafta büyük ve koyu gözlük var mı? (vesikalık için uygun değil!)
    if is_sunglasses_present(file_location):
        return JSONResponse({
            "status": "error",
            "detail": "Fotoğrafta koyu renkli gözlük (güneş gözlüğü) tespit edildi. Vesikalık için gözler açık ve görünür olmalı."
        })

    
    # --- Tüm kontroller geçtiyse vesikalık uygunluğu bildir ---
    return JSONResponse({
        "status": "ok",
        "file_path": file_location,                  # Sunucudaki dosya yolu
        "face_ratio": face_info["face_ratio"],       # Yüz oranı (büyüklüğü)
        "face_offset_x": face_info["offset_x"],      # Yüzün yatay merkeze yakınlığı
        "face_offset_y": face_info["offset_y"],      # Yüzün dikey merkeze yakınlığı
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
        return False  # Fotoğraf açılamazsa boydan sayma

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


