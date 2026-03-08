from flask import Flask, render_template, Response
import cv2, face_recognition, numpy as np, csv, os
from datetime import datetime
app = Flask(__name__)

known_faces = {
    "Ankita": "known_faces/student3.jpg",
    "Yogi Adityanath": "known_faces/image3.png",
    "Modi ji": "known_faces/image2.png",
    "Abdul Kalam ji ": "known_faces/image1.png"
}
known_face_encodings, known_face_names = [], []
for name, path in known_faces.items():
    if os.path.exists(path):
        enc = face_recognition.face_encodings(face_recognition.load_image_file(path))
        if enc: known_face_encodings.append(enc[0]); known_face_names.append(name)
os.makedirs("Attendance", exist_ok=True)
file_path = f"Attendance/{datetime.now():%Y-%m-%d}.csv"
attendance_set = set()
if not os.path.exists(file_path):
    with open(file_path, 'w', newline='') as f: csv.writer(f).writerow(["Name", "Time"])

cap = cv2.VideoCapture(0)
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            name = "Unknown"
            if known_face_encodings:
                dists = face_recognition.face_distance(known_face_encodings, enc)
                best = np.argmin(dists)
                if dists[best] < 0.45: name = known_face_names[best]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if name != "Unknown" and name not in attendance_set:
                attendance_set.add(name)
                with open(file_path, 'a', newline='') as f:
                    csv.writer(f).writerow([name, datetime.now().strftime("%H:%M:%S")])
                    print(f"{name} marked present at {datetime.now():%H:%M:%S}")
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/attendance')
def attendance():
    return render_template('index.html')
@app.route('/records')
def view_records():
    records = []
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_file = f"Attendance/{today_date}.csv"
    if os.path.exists(today_file):
        with open(today_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Name"]:
                    records.append(row)
    return render_template('records.html', records=records, current_date=today_date)
@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/shutdown')
def shutdown(): cap.release(); return "Camera released"
if __name__ == '__main__':
    app.run(debug=True)
