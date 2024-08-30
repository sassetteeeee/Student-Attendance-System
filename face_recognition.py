import tkinter as tk
from tkinter import filedialog, messagebox
import firebase_admin
from firebase_admin import credentials, db
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}  # Dictionary to map label strings to integer IDs
    label_id = 0
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert color image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Extract label from filename by removing the extension
            label = os.path.splitext(filename)[0]
            if label not in label_dict:
                label_dict[label] = label_id
                label_id += 1
            labels.append(label_dict[label])
            images.append(gray_img)
    return images, labels

# Path to the dataset folder
dataset_folder = r'C:\Users\user\PycharmProjects\pythonProject2\Images'

# Load images and labels from dataset
images, labels = load_images_from_folder(dataset_folder)

# Create LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the face recognizer
face_recognizer.train(images, np.array(labels))

# Function to detect and recognize faces
def detect_and_recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(roi_gray)
        if confidence < 70:  # You can adjust the confidence threshold as needed
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return image

class StudentAttendanceSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Student Attendance System")

        # Resize the window
        self.master.geometry("1920x1080")  # Set the window size to 1920x1080 pixels

        # Load the background image
        self.bg_image = Image.open(r"C:\Users\user\PycharmProjects\pythonProject2\Images\bg.jpg")
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.home_frame = tk.Frame(master)
        self.home_frame.pack(fill=tk.BOTH, expand=True)

        # Create a label to display the background image
        self.bg_label = tk.Label(self.home_frame, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.register_frame = tk.Frame(master)
        self.mark_attendance_frame = tk.Frame(master)
        self.view_attendance_frame = tk.Frame(master)

        self.create_home_widgets()
        self.create_register_widgets()
        self.create_mark_attendance_widgets()
        self.create_view_attendance_widgets()

        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(r"C:\Users\user\PycharmProjects\pythonProject2\Resources\student-attendance-syste-80040-firebase-adminsdk-3b3cw-2297ed9199.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://student-attendance-syste-80040-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })

        # Initialize face recognizer and load pre-trained data
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists('trainer.yml'):
            self.recognizer.read('trainer.yml')

    def create_home_widgets(self):
        self.home_label = tk.Label(self.home_frame, text="Welcome to Student Attendance System", font=("Helvetica", 16))
        self.home_label.pack(pady=20)

        self.register_button = tk.Button(self.home_frame, text="Register Student", command=self.show_register_frame)
        self.register_button.pack()

        self.mark_attendance_button = tk.Button(self.home_frame, text="Mark Attendance", command=self.show_mark_attendance_frame)
        self.mark_attendance_button.pack()

        self.view_attendance_button = tk.Button(self.home_frame, text="View Attendance", command=self.show_view_attendance_frame)
        self.view_attendance_button.pack()

    def create_register_widgets(self):
        self.name_label = tk.Label(self.register_frame, text="Name:")
        self.name_entry = tk.Entry(self.register_frame)
        self.name_label.pack()
        self.name_entry.pack()

        self.matric_label = tk.Label(self.register_frame, text="Matric Number:")
        self.matric_entry = tk.Entry(self.register_frame)
        self.matric_label.pack()
        self.matric_entry.pack()

        self.year_label = tk.Label(self.register_frame, text="Year:")
        self.year_entry = tk.Entry(self.register_frame)
        self.year_label.pack()
        self.year_entry.pack()

        self.section_label = tk.Label(self.register_frame, text="Section:")
        self.section_entry = tk.Entry(self.register_frame)
        self.section_label.pack()
        self.section_entry.pack()

        self.subject_label = tk.Label(self.register_frame, text="Subject:")
        self.subject_entry = tk.Entry(self.register_frame)
        self.subject_label.pack()
        self.subject_entry.pack()

        self.faculty_label = tk.Label(self.register_frame, text="Faculty:")
        self.faculty_entry = tk.Entry(self.register_frame)
        self.faculty_label.pack()
        self.faculty_entry.pack()

        self.phone_label = tk.Label(self.register_frame, text="Phone Number:")
        self.phone_entry = tk.Entry(self.register_frame)
        self.phone_label.pack()
        self.phone_entry.pack()

        self.photo_label = tk.Label(self.register_frame, text="Upload Photo:")
        self.upload_button = tk.Button(self.register_frame, text="Upload", command=self.upload_photo)
        self.photo_label.pack()
        self.upload_button.pack()

        self.register_button = tk.Button(self.register_frame, text="Register", command=self.register_student)
        self.register_button.pack()

        self.home_button_from_register = tk.Button(self.register_frame, text="Back to Home", command=self.show_home_frame)
        self.home_button_from_register.pack()

    def create_mark_attendance_widgets(self):
        self.mark_attendance_label = tk.Label(self.mark_attendance_frame, text="Mark Attendance", font=("Helvetica", 16))
        self.mark_attendance_label.pack(pady=20)

        self.start_camera_button = tk.Button(self.mark_attendance_frame, text="Start Camera", command=self.start_camera)
        self.start_camera_button.pack()

        self.home_button_from_mark_attendance = tk.Button(self.mark_attendance_frame, text="Back to Home", command=self.show_home_frame)
        self.home_button_from_mark_attendance.pack()

    def create_view_attendance_widgets(self):
        self.view_attendance_label = tk.Label(self.view_attendance_frame, text="View Attendance", font=("Helvetica", 16))
        self.view_attendance_label.pack(pady=20)

        self.view_button = tk.Button(self.view_attendance_frame, text="View Attendance", command=self.view_attendance)
        self.view_button.pack()

        self.home_button_from_view_attendance = tk.Button(self.view_attendance_frame, text="Back to Home", command=self.show_home_frame)
        self.home_button_from_view_attendance.pack()

    def show_home_frame(self):
        self.home_frame.pack()
        self.register_frame.pack_forget()
        self.mark_attendance_frame.pack_forget()
        self.view_attendance_frame.pack_forget()

    def show_register_frame(self):
        self.register_frame.pack()
        self.home_frame.pack_forget()
        self.mark_attendance_frame.pack_forget()
        self.view_attendance_frame.pack_forget()

    def show_mark_attendance_frame(self):
        self.mark_attendance_frame.pack()
        self.home_frame.pack_forget()
        self.register_frame.pack_forget()
        self.view_attendance_frame.pack_forget()

    def show_view_attendance_frame(self):
        self.view_attendance_frame.pack()
        self.home_frame.pack_forget()
        self.register_frame.pack_forget()
        self.mark_attendance_frame.pack_forget()

    def upload_photo(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            print("Uploaded Photo Path:", self.file_path)
        else:
            print("No file selected.")

    def register_student(self):
        name = self.name_entry.get()
        matric_number = self.matric_entry.get()
        year = self.year_entry.get()
        section = self.section_entry.get()
        subject = self.subject_entry.get()
        faculty = self.faculty_entry.get()
        phone_number = self.phone_entry.get()

        if not all([name, matric_number, year, section, subject, faculty, phone_number, self.file_path]):
            messagebox.showerror("Error", "All fields and photo must be filled")
            return

        student_ref = db.reference('students')
        student_data = {
            'name': name,
            'matric number': matric_number,
            'year': year,
            'section': section,
            'subject': subject,
            'faculty': faculty,
            'phone number': phone_number
        }
        student_ref.push(student_data)

        face_id = matric_number
        self.save_face(face_id, self.file_path)

        messagebox.showinfo("Success", "Registration successful!")
        self.train_recognizer()

    def save_face(self, face_id, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if not os.path.exists("dataset"):
                os.makedirs("dataset")
            cv2.imwrite(f"dataset/User.{face_id}.{str(datetime.now().strftime('%Y%m%d%H%M%S'))}.jpg", gray[y:y + h, x:x + w])

    def train_recognizer(self):
        path = 'dataset'
        if not os.path.exists(path):
            os.makedirs(path)

        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            img = Image.open(image_path).convert('L')
            img_numpy = np.array(img, 'uint8')
            id_ = int(os.path.split(image_path)[-1].split('.')[1])
            faces = self.face_cascade.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id_)

        self.recognizer.train(face_samples, np.array(ids))
        self.recognizer.write('trainer.yml')

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                break

            frame = detect_and_recognize_faces(frame)
            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def mark_attendance(self, student_info):
        with open('attendance.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [student_info['matric number'], student_info['name'], student_info['year'], student_info['section'],
                 str(datetime.now())])
        print(f"Attendance marked for {student_info['name']}")

    def view_attendance(self):
        if not os.path.exists('attendance.csv'):
            messagebox.showerror("Error", "No attendance records found")
            return

        df = pd.read_csv('attendance.csv', names=['Matric Number', 'Name', 'Year', 'Section', 'Timestamp'])
        attendance_count = df['Name'].value_counts()
        attendance_count.plot(kind='bar')
        plt.title('Attendance Count per Student')
        plt.xlabel('Student Name')
        plt.ylabel('Attendance Count')
        plt.show()

def main():
    root = tk.Tk()
    app = StudentAttendanceSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()
