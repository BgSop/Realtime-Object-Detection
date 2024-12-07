import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load classifier untuk mendeteksi wajah, mata, dan senyum
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Fungsi untuk mendeteksi wajah
def detect_face(img):
    if img is None:
        return None  # Menghindari error jika gambar kosong
    
    face_img = img.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = face_img[y:y + h, x:x + w]

        # Deteksi mata dan senyum dalam wajah
        detect_eyes(roi_gray, roi_color)
        detect_smile(roi_gray, roi_color)

    return face_img

# Fungsi untuk mendeteksi mata
def detect_eyes(roi_gray, roi_color):
    eyes_rects = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes_rects:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Fungsi untuk mendeteksi senyum
def detect_smile(roi_gray, roi_color):
    smile_rects = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
    for (sx, sy, sw, sh) in smile_rects:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

# Membuka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka kamera!")
    exit()  # Jika kamera tidak terbuka, keluar dari program

# Loop untuk membaca frame dari kamera
while True:
    ret, frame = cap.read()  # Membaca frame dari kamera

    if not ret or frame is None:
        print("Gagal membaca frame!")
        break  # Keluar dari loop jika pembacaan frame gagal

    # Deteksi wajah, mata, dan senyum
    frame = detect_face(frame)

    # Menambahkan teks di layar
    cv2.putText(frame, "Tekan ESC untuk keluar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Menampilkan hasil deteksi pada video
    cv2.imshow('Video Face Detect', frame)

    # Menunggu input dari pengguna untuk keluar (tekan ESC untuk keluar)
    k = cv2.waitKey(1)
    if k == 27:  # ESC untuk keluar
        break

# Melepaskan objek kamera dan menutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()

# Melepaskan objek kamera dan menutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()