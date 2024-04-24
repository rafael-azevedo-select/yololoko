from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import torch
import os
from datetime import datetime
import sqlite3

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/var/www/yololoko/html/uploads/'

db_connection = sqlite3.connect('detections.db', check_same_thread=False)
db_cursor = db_connection.cursor()

db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        weapon_type TEXT,
        image_filename TEXT
    )
''')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'missing file'}), 400
    file = request.files['image']
    filename = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    results = model(image)
    detected_objects = [results.names[int(cls)] for cls in results.xyxy[0][:, -1]]

    if any(obj in ['pistol', 'knife'] for obj in detected_objects):
        db_cursor.execute("INSERT INTO detections (weapon_type, image_filename) VALUES (?, ?)", (', '.join(detected_objects), filename))
        db_connection.commit()

    imagePath = os.path.join('uploads', filename)
    return jsonify({'detected_objects': detected_objects, 'imagePath': imagePath})

@app.route('/detections', methods=['GET'])
def show_detections():
    db_cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 5")
    detections = db_cursor.fetchall()
    return render_template('detections.html', detections=detections)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

