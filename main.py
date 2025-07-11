# main.py - Face Recognition & Attendance Logging (No Location Column)
import cv2
import os
import pandas as pd
from datetime import datetime

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists("trained_model.yml"):
    print("❌ Model not found. Please train the model first.")
    exit()
recognizer.read("trained_model.yml")

# Load student ID and name mapping
label_to_identity = {}
for folder in os.listdir("face_dataset"):
    try:
        label = int(folder.split('_')[0])
        name = folder.split('_')[1]
        label_to_identity[label] = (str(label), name)
    except:
        continue

# Attendance CSV
FILE_NAME = "attendance.csv"

# Ensure the attendance CSV file exists
if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    df.to_csv(FILE_NAME, index=False)

def mark_attendance(id_, name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    try:
        df = pd.read_csv(FILE_NAME)
    except PermissionError:
        print("❌ Please close 'attendance.csv' if it's open in Excel or another program.")
        return

    if not ((df['ID'] == id_) & (df['Date'] == date)).any():
        df.loc[len(df)] = [id_, name, date, time]
        df.to_csv(FILE_NAME, index=False)
        print(f"✅ Marked: {name} ({id_}) at {time}")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

marked_ids = []
print("🟢 Starting Smart Attendance System. Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

        label, confidence = recognizer.predict(roi)

        if confidence < 60:
            id_, name = label_to_identity.get(label, ("?", "Unknown"))
            if id_ not in marked_ids:
                mark_attendance(id_, name)
                marked_ids.append(id_)
            label_text = f"{name} ({id_})"
        else:
            label_text = "Unknown"

        # Always draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("📌 System closed. All attendance marked successfully.")
