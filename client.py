import time
import os
import requests
import shutil
from datetime import datetime
from PIL import Image

SHOW_IMAGES = False #opsiyonel olarak görüntüleri gösterir
# Eğer True ise, API'den gelen resimler ekranda gösterilir.
# Eğer False ise, sadece dosyalar uygun/uygun değil olarak işlenir.


# Aynı isimde dosya varsa üzerine yazmamak için benzersiz dosya adı üretir
def benzersiz_dosya_yolu(dosya_yolu):
    if not os.path.exists(dosya_yolu):
        return dosya_yolu

    dosya_adi, uzanti = os.path.splitext(dosya_yolu)
    sayac = 1
    while True:
        yeni_dosya_yolu = f"{dosya_adi}_{sayac}{uzanti}"
        if not os.path.exists(yeni_dosya_yolu):
            return yeni_dosya_yolu
        sayac += 1


# Tarihe göre klasör oluşturmak için bugünün tarihi alınır
BUGUN = datetime.now().strftime("%Y-%m-%d")

# Uygun ve Red klasörlerinin içine tarihli alt klasörler tanımlanır
KONTROL_KLASORU = "Kontrol"
UYGUN_KLASORU = os.path.join("Uygun", BUGUN)
RED_KLASORU = os.path.join("Red", BUGUN)

# Klasörler yoksa oluşturulur (varsa hata vermez)
os.makedirs(RED_KLASORU, exist_ok=True)
os.makedirs(UYGUN_KLASORU, exist_ok=True)

API_URL = "http://localhost:8000/upload"  # API sunucun burada çalışıyorsa


# İstatistik sayaçları
uygun_sayisi = 0
red_sayisi = 0

try:
    while True:
        # Her döngüde yeni dosya var mı diye kontrol et
        dosya_listesi = os.listdir(KONTROL_KLASORU)
        if not dosya_listesi:
            time.sleep(10)
            continue
        else:
            for dosya_adi in dosya_listesi:
                dosya_yolu = os.path.join(KONTROL_KLASORU, dosya_adi)

                if not os.path.isfile(dosya_yolu):
                    continue

                try:
                    img = Image.open(dosya_yolu)
                    if SHOW_IMAGES:
                        img.show()
                except Exception as e:
                    print(f"[!] {dosya_adi} görüntülenemedi: {e}")

                with open(dosya_yolu, "rb") as f:
                    files = {"file": (dosya_adi, f, "image/jpeg")}
                    try:
                        response = requests.post(API_URL, files=files)
                        response.raise_for_status()
                    except Exception as e:
                        print(f"[!] {dosya_adi} için API hatası: {e}")
                        continue

                veri = response.json()

                if veri.get("status") == "ok":
                    # Aynı isimli dosya varsa üzerine yazmamak için hedef yol oluşturulur
                    hedef_yol = benzersiz_dosya_yolu(
                        os.path.join(UYGUN_KLASORU, dosya_adi))
                    shutil.copy2(dosya_yolu, hedef_yol)
                    mesaj = f"{datetime.now()} - [✓] {dosya_adi} uygun → '{hedef_yol}' klasörüne kopyalandı."
                    print(mesaj)
                    with open("client_logs.txt", "a") as log:
                        log.write(mesaj + "\n")
                    # artık işimiz kalmayan Kontrol klasöründeki orijinal dosyayı siliyor.
                    os.remove(dosya_yolu)
                    uygun_sayisi += 1
                # Eğer uygun değilse, red klasörüne kopyala
                else:
                    # Aynı isimli dosya varsa üzerine yazmamak için hedef yol oluşturulur
                    hedef_yol = benzersiz_dosya_yolu(
                        os.path.join(RED_KLASORU, dosya_adi))
                    shutil.copy2(dosya_yolu, hedef_yol)  # Red klasörüne kopyala
                    sebep = (
                        (veri.get("errors") or [{}])[0].get("msg") if isinstance(veri.get("errors"), list) else None
                    ) or veri.get("msg") or veri.get("message") or "Sebep belirtilmedi."
                    mesaj = f"{datetime.now()} - [✗] {dosya_adi} uygun değil. Sebep: {sebep}"
                    print(mesaj)
                    with open("client_logs.txt", "a") as log:  # log kaydı oluştur
                        log.write(mesaj + "\n")
                    os.remove(dosya_yolu)
                    red_sayisi += 1

        time.sleep(10)
except KeyboardInterrupt:
    pass
