from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import requests
import base64
from dotenv import load_dotenv
import io
from flask import send_file

# Load API key from .env file
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

ROBOFLOW_MODELS = {
    "model_1": {
        "id": "soldier-ijybv/2 ",
        "use_case": "For soldier detection",
        "confidence_threshold": 0.5
    },
    "model_2": {
        "id": "​​military-f5tbj/1",
        "use_case": "For military equipments detection",
        "confidence_threshold": 0.5
    },
    "model_3": {
        "id": "tank-2xykr/3",
        "use_case": "For tank detection",
        "confidence_threshold": 0.5
    },
    "model_4": {
        "id": "fighter-jet-detection/1",
        "use_case": "For fighter jet detection",
        "confidence_threshold": 0.5
    },
    "model_5": {
        "id": "guns-miu7v/1",
        "use_case": "For gun detection",
        "confidence_threshold": 0.5
    },
    "model_6": {
        "id": "weapon-detection-t2esr/1",
        "use_case": "For weapon detection",
        "confidence_threshold": 0.5
    },
    "model_7": {
        "id": "mine-xamdv/1",
        "use_case": "For mine detection",
        "confidence_threshold": 0.5
    },
    "model_8": {
        "id": "landmine-detection-vsmzn/1",
        "use_case": "For landmine detection",
        "confidence_threshold": 0.5
    }
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    preview_img = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', models=ROBOFLOW_MODELS, error='No image uploaded')
        file = request.files['image']
        if file.filename == '':
            return render_template('upload.html', models=ROBOFLOW_MODELS, error='No image selected')
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return render_template('upload.html', models=ROBOFLOW_MODELS, error='Invalid image file')
        # For preview, keep a copy
        preview_img = image.copy()
        results = {}
        for model_key, model_info in ROBOFLOW_MODELS.items():
            url = f"https://detect.roboflow.com/{model_info['id']}?api_key={ROBOFLOW_API_KEY}"
            params = {
                "confidence": model_info["confidence_threshold"],
                "overlap": 30,
                "format": "json"
            }
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response = requests.post(
                url,
                params=params,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            if response.status_code == 200:
                predictions = response.json().get('predictions', [])
                results[model_key] = []
                for pred in predictions:
                    x_center = pred.get('x', 0)
                    y_center = pred.get('y', 0)
                    width = pred.get('width', 0)
                    height = pred.get('height', 0)
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    class_name = pred.get('class', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    results[model_key].append({'class': class_name, 'confidence': confidence})
                    # Draw rectangle and label on preview_img
                    cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(preview_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(preview_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                results[model_key] = []
        # Encode preview_img for display
        _, preview_buffer = cv2.imencode('.jpg', preview_img)
        preview_base64 = base64.b64encode(preview_buffer).decode('utf-8')
        return render_template('upload.html', models=ROBOFLOW_MODELS, results=results, image_uploaded=True, preview_img=preview_base64)
    return render_template('upload.html', models=ROBOFLOW_MODELS)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
