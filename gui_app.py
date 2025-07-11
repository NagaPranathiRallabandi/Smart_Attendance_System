# gui_app.py — Fullscreen GUI with Attendance Viewer & Export Feature

import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
import os
import cv2
import pandas as pd
from datetime import datetime

# Ensure required folders exist
if not os.path.exists("face_dataset"):
    os.makedirs("face_dataset")

REGISTRATION_LOG = "registered_users.csv"

# Initialize CSV if not exists
def init_csv():
    if not os.path.exists(REGISTRATION_LOG):
        df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
        df.to_csv(REGISTRATION_LOG, index=False)

# Save registration details to CSV
def log_registration(id_, name):
    init_csv()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    df = pd.read_csv(REGISTRATION_LOG)
    df.loc[len(df.index)] = [id_, name, date, time]
    df.to_csv(REGISTRATION_LOG, index=False)

# Register Person (Capture 20 photos)
def register_person():
    name = name_entry.get().strip()
    id_ = id_entry.get().strip()
    if not name or not id_:
        messagebox.showerror("Error", "Please enter both Name and ID.")
        return

    folder_name = f"face_dataset/{id_}_{name}"
    os.makedirs(folder_name, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    messagebox.showinfo("Info", "Webcam started. Look at the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            filename = f"{folder_name}/{count}.jpg"
            cv2.imwrite(filename, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Registering Face - ESC to Exit", frame)
        if cv2.waitKey(1) == 27 or count >= 40:
            break

    cap.release()
    cv2.destroyAllWindows()
    log_registration(id_, name)
    messagebox.showinfo("Done", f"Captured {count} images for {name}.")

# Run trainer.py
def train_model():
    subprocess.run(["python", "trainer.py"])
    messagebox.showinfo("Trained", "Model trained successfully!")

# Run main.py for live detection
def start_attendance():
    subprocess.Popen(["python", "main.py"])

# View today's attendance
def show_attendance():
    try:
        df = pd.read_csv("attendance.csv")
        today = datetime.now().strftime("%Y-%m-%d")
        df_today = df[df['Date'] == today]

        if df_today.empty:
            messagebox.showinfo("Attendance", "No attendance marked today.")
            return

        top = tk.Toplevel(root)
        top.title("Today's Attendance")
        top.geometry("600x400")
        top.configure(bg="#2d3436")

        text = tk.Text(top, wrap='none', font=("Arial", 12), bg="#dfe6e9", fg="#2d3436")
        text.pack(fill="both", expand=True)
        text.insert("end", df_today.to_string(index=False))
    except Exception as e:
        messagebox.showerror("Error", f"Could not load attendance: {str(e)}")

# Export full attendance CSV
def export_attendance():
    try:
        df = pd.read_csv("attendance.csv")
        export_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if export_path:
            df.to_csv(export_path, index=False)
            messagebox.showinfo("Exported", f"Attendance exported to {export_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Export failed: {str(e)}")

# GUI Setup
root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("900x650")
root.configure(bg="#1e272e")

# Header
header = tk.Label(root, text="🎓 Smart Attendance System", font=("Arial Rounded MT Bold", 30, "bold"), bg="#1e272e", fg="#ffffff")
header.pack(pady=30)

# Form Frame
form_frame = tk.Frame(root, bg="#1e272e")
form_frame.pack(pady=10)

label_style = {"font": ("Arial", 16, "bold"), "bg": "#1e272e", "fg": "#d2dae2"}
entry_style = {"font": ("Arial", 16), "width": 30, "bg": "#dcdde1", "fg": "#2f3640"}

# Name input
tk.Label(form_frame, text="Student Name:", **label_style).grid(row=0, column=0, pady=10, sticky="e")
name_entry = tk.Entry(form_frame, **entry_style)
name_entry.grid(row=0, column=1, pady=10, padx=10)

# ID input
tk.Label(form_frame, text="Student ID:", **label_style).grid(row=1, column=0, pady=10, sticky="e")
id_entry = tk.Entry(form_frame, **entry_style)
id_entry.grid(row=1, column=1, pady=10, padx=10)

# Button Frame
button_frame = tk.Frame(root, bg="#1e272e")
button_frame.pack(pady=40)

btn_style = {"font": ("Arial", 14, "bold"), "width": 30, "padx": 5, "pady": 10, "bd": 0, "relief": "ridge"}

# Buttons
register_btn = tk.Button(button_frame, text="📸 Register Face", bg="#0984e3", fg="white", command=register_person, **btn_style)
register_btn.pack(pady=10)

train_btn = tk.Button(button_frame, text="🧠 Train Model", bg="#00b894", fg="white", command=train_model, **btn_style)
train_btn.pack(pady=10)

start_btn = tk.Button(button_frame, text="🟢 Start Attendance", bg="#fdcb6e", fg="#2d3436", command=start_attendance, **btn_style)
start_btn.pack(pady=10)

view_attendance_btn = tk.Button(button_frame, text="📋 View Today's Attendance", bg="#6c5ce7", fg="white", command=show_attendance, **btn_style)
view_attendance_btn.pack(pady=10)

export_btn = tk.Button(button_frame, text="📤 Export Attendance CSV", bg="#00cec9", fg="white", command=export_attendance, **btn_style)
export_btn.pack(pady=10)

# Footer
footer = tk.Label(root, text="Developed by Ramya & Pranathi", font=("Arial", 10), bg="#1e272e", fg="#a4b0be")
footer.pack(side="bottom", pady=15)

# Run GUI
root.mainloop()
