# trainer.py — Train model from face_dataset folder
import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = 'face_dataset'
faces = []
labels = []

print("🔍 Loading images from dataset...")

for folder in os.listdir(dataset_path):
    try:
        label = int(folder.split('_')[0])
    except:
        print(f"⚠️ Skipping invalid folder: {folder}")
        continue

    person_path = os.path.join(dataset_path, folder)
    for image_file in os.listdir(person_path):
        img_path = os.path.join(person_path, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(label)
            print(f"✅ Loaded: {img_path}")

if not faces:
    print("❌ No valid training images found. Please register a person first.")
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save("trained_model.yml")
print("✅ Model trained and saved as trained_model.yml")
