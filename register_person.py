import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

name = input("Enter full name: ")
id = input("Enter numeric ID: ")
folder_name = f"face_dataset/{id}_{name}"
os.makedirs(folder_name, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("📸 Capturing 10 face images. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        img_path = f"{folder_name}/{count}.jpg"
        cv2.imwrite(img_path, face)
        count += 1
        print(f"✅ Saved {img_path}")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Registering Face", frame)
    if cv2.waitKey(1) == 27 or count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print(f"📁 Saved {count} face images to {folder_name}")
print("📌 Now run trainer.py once to update the recognition model.")
