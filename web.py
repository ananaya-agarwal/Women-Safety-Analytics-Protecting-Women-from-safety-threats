from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load models
yolo_model = YOLO('yolov8n.pt')  # Person detection
gender_model = load_model('gender_model.h5')  # Gender classifier (dummy/demo)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Simple page with camera upload

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run YOLO Detection
    results = yolo_model(filepath)
    image = cv2.imread(filepath)
    men, women = 0, 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, cls in zip(boxes, classes):
            if int(cls) == 0:  # Class 0 = Person
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue  # Skip empty crops
                
                # Preprocess for gender classifier
                resized_crop = cv2.resize(crop, (64, 64)) / 255.0
                resized_crop = np.expand_dims(resized_crop, axis=0)
                pred = gender_model.predict(resized_crop, verbose=0)
                
                gender = 'Male' if pred[0][0] > 0.5 else 'Female'
                color = (255, 0, 0) if gender == 'Male' else (0, 0, 255)
                
                # Count genders
                if gender == 'Male':
                    men += 1
                else:
                    women += 1

                # Draw bounding box & label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Save annotated image
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, image)

    return jsonify({
        'men': men,
        'women': women,
        'result_image_url': f'/static/outputs/{filename}'
    })

@app.route('/static/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
