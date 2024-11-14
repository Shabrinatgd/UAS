import cv2
import os
import time

# Menangkap video dari webcam (gunakan 0 untuk webcam default)
cap = cv2.VideoCapture(0)

# Pastikan webcam terbuka dengan benar
if not cap.isOpened():
    print("Error: Webcam tidak ditemukan.")
    exit()

# Ambil gambar satu kali
ret, frame = cap.read()

# Jika gambar berhasil diambil
if ret:
    # Deteksi wajah dalam gambar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Buat nama folder berdasarkan waktu atau nama foto
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"foto_{timestamp}"

    # Membuat folder untuk menyimpan gambar jika belum ada
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for (x, y, w, h) in faces:
        # Menandai wajah yang terdeteksi (untuk pengujian)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Menyimpan gambar original
        cv2.imwrite(os.path.join(folder_name, "original.jpg"), frame)

        # Crop wajah dari gambar original
        face_cropped = frame[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(folder_name, "cropped_face.jpg"), face_cropped)

        # Convert ke grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(folder_name, "grayscale.jpg"), gray_image)

        # Convert ke black and white (thresholding)
        _, bw_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(folder_name, "blackwhite.jpg"), bw_image)

        # Menampilkan gambar yang telah diambil
        cv2.imshow("Captured Image", frame)
        cv2.imshow("Cropped Face", face_cropped)

# Tunggu beberapa detik agar gambar bisa dilihat
cv2.waitKey(2000)

# Menutup webcam dan jendela
cap.release()
cv2.destroyAllWindows()

print("Gambar telah disimpan di folder:", folder_name)
