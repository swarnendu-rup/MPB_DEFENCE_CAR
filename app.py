"""
Flask Object Detection with Image Upload and Roboflow Models
Modified to use image upload for detection with Flask web interface
"""

import cv2
import numpy as np
import requests
import json
import base64
import time
from flask import Flask, render_template, request, jsonify, Response
import threading
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Model Configuration - Add your 8 model IDs here
# These are public/pre-trained models that don't require API keys
ROBOFLOW_MODELS = {
    # TODO: Replace these placeholder model IDs with your actual Roboflow model IDs
    "model_1": {
        "id": "soldier-ijybv/2 ",  # Replace with actual model ID (e.g., "yolov8n-640")
        "use_case": "For soldier detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_2": {
        "id": "​​military-f5tbj/1",  # Replace with actual model ID
        "use_case": "For military equipments detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_3": {
        "id": "tank-2xykr/3",  # Replace with actual model ID
        "use_case": "For tank detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_4": {
        "id": "fighter-jet-detection/1",  # Replace with actual model ID
        "use_case": "For fighter jet detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_5": {
        "id": "guns-miu7v/1",  # Replace with actual model ID
        "use_case": "For gun detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_6": {
        "id": "weapon-detection-t2esr/1",  # Replace with actual model ID
        "use_case": "For weapon detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_7": {
        "id": "mine-xamdv/1",  # Replace with actual model ID
        "use_case": "For mine detection",  # Add description
        "confidence_threshold": 0.5
    },
    "model_8": {
        "id": "landmine-detection-vsmzn/1",  # Replace with actual model ID
        "use_case": "For landmine detection",  # Add description
        "confidence_threshold": 0.5
    },
    # Add combined models for UI selection
    "model_5_6": {
        "id": "combined-gun-weapon",
        "use_case": "For gun and weapon detection (model 5 & 6)",
        "confidence_threshold": 0.5
    },
    "model_7_8": {
        "id": "combined-mine-landmine",
        "use_case": "For mine and landmine detection (model 7 & 8)",
        "confidence_threshold": 0.5
    }
}

# Flask app
app = Flask(__name__)

# Global variables
detector = None
current_model = "model_1"
detection_enabled = True
latest_detections = []

class RoboflowDetector:
    """
    Roboflow object detection class using public models (no API key required)
    """
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.current_model = current_model
        
    def encode_image_to_base64(self, image):
        """
        Convert OpenCV image to base64 string for Roboflow API
        """
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def detect_objects(self, image):
        """
        Perform object detection using Roboflow public models
        """
        try:
            # Get current model configuration
            model_info = self.model_config[self.current_model]
            model_id = model_info["id"]
            confidence = model_info["confidence_threshold"]
            
            # Encode image to base64
            img_base64 = self.encode_image_to_base64(image)
            
            # Roboflow inference API endpoint with API key
            url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
            
            # Parameters for public model inference
            params = {
                "confidence": confidence,
                "overlap": 30,  # Non-maximum suppression threshold
                "format": "json"
            }
            
            # Send request to Roboflow
            response = requests.post(
                url,
                params=params,
                data=img_base64,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            if response.status_code == 200:
                predictions = response.json()
                return self.process_predictions(predictions, image)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return image, []
                
        except Exception as e:
            print(f"Detection error: {e}")
            return image, []
    
    def process_predictions(self, predictions, image):
        """
        Process Roboflow predictions and draw bounding boxes
        """
        detections = []
        
        if 'predictions' in predictions:
            for prediction in predictions['predictions']:
                # Extract detection information
                class_name = prediction.get('class', 'Unknown')
                confidence = prediction.get('confidence', 0)
                
                # Get bounding box coordinates
                x_center = prediction.get('x', 0)
                y_center = prediction.get('y', 0)
                width = prediction.get('width', 0)
                height = prediction.get('height', 0)
                
                # Convert to OpenCV format (top-left corner)
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Store detection info
                detection_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                }
                detections.append(detection_info)
                
                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image, detections
    
    def switch_model(self, model_key):
        """
        Switch to a different Roboflow model
        """
        if model_key in self.model_config:
            self.current_model = model_key
            model_info = self.model_config[model_key]
            print(f"Switched to model: {model_key}")
            print(f"Use case: {model_info['use_case']}")
            print(f"Model ID: {model_info['id']}")
            return True
        else:
            print(f"Model {model_key} not found in configuration")
            return False
    
    def detect_objects_with_model(self, image, model_key):
        """
        Detect objects using a specific model
        """
        try:
            model_info = self.model_config[model_key]
            model_id = model_info["id"]
            confidence = model_info["confidence_threshold"]
            img_base64 = self.encode_image_to_base64(image)
            url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
            params = {
                "confidence": confidence,
                "overlap": 30,
                "format": "json"
            }
            response = requests.post(
                url,
                params=params,
                data=img_base64,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            if response.status_code == 200:
                predictions = response.json()
                return self.process_predictions(predictions, image)
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return image, []
        except Exception as e:
            print(f"Detection error: {e}")
            return image, []

class CameraManager:
    """
    Manages laptop camera capture
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.initialize_camera()
    def initialize_camera(self):
        """
        Initialize the laptop camera
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Camera {self.camera_index} initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    def get_frame(self):
        """
        Get a frame from the camera
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
    def release(self):
        """
        Release the camera
        """
        if self.cap is not None:
            self.cap.release()

def generate_frames():
    """
    Generate video frames for Flask streaming
    """
    global camera, detector, detection_enabled, latest_detections, current_model
    while True:
        if camera is None:
            break
        frame = camera.get_frame()
        if frame is None:
            continue
        detections = []
        detected_frame = frame.copy()
        # Perform object detection if enabled
        if detection_enabled and detector is not None:
            detected_frame, detections = detector.detect_objects(frame.copy())
            latest_detections = detections
        # Add model info to frame
        model_info = ROBOFLOW_MODELS.get(current_model, {"use_case": "Detection"})
        info_text = f"Model: {current_model} | Use case: {model_info['use_case']}"
        cv2.putText(detected_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    Video streaming route
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main page for image upload and detection results
    """
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', models=ROBOFLOW_MODELS, error='No image uploaded')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', models=ROBOFLOW_MODELS, error='No image selected')
        # Read image as numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return render_template('index.html', models=ROBOFLOW_MODELS, error='Invalid image file')
        # Run detection for all models
        detector = RoboflowDetector(ROBOFLOW_MODELS)
        results = {}
        for model_key in ROBOFLOW_MODELS.keys():
            detected_img, detections = detector.detect_objects_with_model(image.copy(), model_key)
            results[model_key] = detections
        return render_template('index.html', models=ROBOFLOW_MODELS, results=results, image_uploaded=True)
    return render_template('index.html', models=ROBOFLOW_MODELS)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """
    Switch between different Roboflow models
    """
    global current_model, detector
    
    data = request.get_json()
    model_key = data.get('model_key')
    
    if model_key in ROBOFLOW_MODELS:
        current_model = model_key
        detector.switch_model(model_key)
        return jsonify({'status': 'success', 'model': model_key})
    else:
        return jsonify({'status': 'error', 'message': 'Model not found'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """
    Toggle object detection on/off
    """
    global detection_enabled
    
    detection_enabled = not detection_enabled
    return jsonify({'status': 'success', 'enabled': detection_enabled})

@app.route('/get_detections')
def get_detections():
    """
    Get current detection results
    """
    return jsonify({'detections': latest_detections})

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    """
    Update confidence threshold for current model
    """
    global detector
    
    data = request.get_json()
    confidence = data.get('confidence', 0.5)
    
    if 0 <= confidence <= 1:
        ROBOFLOW_MODELS[current_model]['confidence_threshold'] = confidence
        return jsonify({'status': 'success', 'confidence': confidence})
    else:
        return jsonify({'status': 'error', 'message': 'Confidence must be between 0 and 1'})

if __name__ == '__main__':
    detector = RoboflowDetector(ROBOFLOW_MODELS)
    camera = None
    # Try multiple camera indices for webcam compatibility
    for cam_index in [0, 1, 2]:
        test_camera = CameraManager(camera_index=cam_index)
        if test_camera.cap is not None and test_camera.cap.isOpened():
            camera = test_camera
            print(f"Camera {cam_index} initialized and opened.")
            break
        else:
            print(f"Camera {cam_index} not available.")
    if camera is None:
        print("No available camera found. Please check your webcam connection or try a different index.")
        exit(1)
    print("Starting Flask Object Detection Server...")
    print("Camera initialized, starting web server...")
    print("Access the application at: http://localhost:5000")
    print(f"Current model: {current_model}")
    print(f"Use case: {ROBOFLOW_MODELS[current_model]['use_case']}")
    try:
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        if camera:
            camera.release()
        print("Server shutting down")

"""
HTML TEMPLATE NEEDED:
Create a folder called 'templates' in the same directory as this script
and create a file called 'index.html' with the following content:

<!DOCTYPE html>
<html>
<head>
    <title>Object Detection with Roboflow</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .video-container { text-align: center; margin-bottom: 30px; }
        .video-stream { border: 3px solid #333; border-radius: 10px; max-width: 100%; height: auto; }
        .controls { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-bottom: 20px; }
        .control-group { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .model-buttons { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .model-btn { padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; transition: all 0.3s; }
        .model-btn.active { background-color: #4CAF50; color: white; }
        .model-btn:not(.active) { background-color: #e0e0e0; }
        .model-btn:hover { opacity: 0.8; }
        .toggle-btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .toggle-btn.enabled { background-color: #4CAF50; color: white; }
        .toggle-btn.disabled { background-color: #f44336; color: white; }
        .confidence-slider { width: 200px; margin: 10px; }
        .detection-info { background: white; padding: 15px; border-radius: 10px; margin-top: 20px; }
        .detection-list { max-height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Object Detection with Roboflow Models</h1>
            <p>Live detection using laptop camera</p>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Camera Feed">
        </div>
        
        <div class="controls">
            <div class="control-group">
                <h3>Model Selection</h3>
                <div class="model-buttons">
                    {% for model_key, model_info in models.items() %}
                    <button class="model-btn {% if model_key == current_model %}active{% endif %}" 
                            onclick="switchModel('{{ model_key }}')">
                        {{ model_key }}<br>
                        <small>{{ model_info.use_case }}</small>
                    </button>
                    {% endfor %}
                </div>
            </div>
            
            <div class="control-group">
                <h3>Detection Control</h3>
                <button id="toggle-detection" class="toggle-btn enabled" onclick="toggleDetection()">
                    Detection ON
                </button>
                <br><br>
                <label>Confidence Threshold: <span id="confidence-value">0.5</span></label><br>
                <input type="range" class="confidence-slider" min="0" max="1" step="0.1" value="0.5" 
                       onchange="updateConfidence(this.value)">
            </div>
        </div>
        
        <div class="detection-info">
            <h3>Current Detections</h3>
            <div id="detection-list" class="detection-list">
                No detections yet...
            </div>
        </div>
    </div>

    <script>
        let detectionEnabled = true;
        let currentModel = '{{ current_model }}';
        
        function switchModel(modelKey) {
            fetch('/switch_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_key: modelKey })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentModel = modelKey;
                    document.querySelectorAll('.model-btn').forEach(btn => btn.classList.remove('active'));
                    event.target.classList.add('active');
                }
            });
        }
        
        function toggleDetection() {
            fetch('/toggle_detection', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                detectionEnabled = data.enabled;
                const btn = document.getElementById('toggle-detection');
                btn.textContent = detectionEnabled ? 'Detection ON' : 'Detection OFF';
                btn.className = detectionEnabled ? 'toggle-btn enabled' : 'toggle-btn disabled';
            });
        }
        
        function updateConfidence(value) {
            document.getElementById('confidence-value').textContent = value;
            fetch('/update_confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ confidence: parseFloat(value) })
            });
        }
        
        function updateDetections() {
            fetch('/get_detections')
            .then(response => response.json())
            .then(data => {
                const detectionList = document.getElementById('detection-list');
                if (data.detections.length === 0) {
                    detectionList.innerHTML = 'No detections...';
                } else {
                    detectionList.innerHTML = data.detections
                        .map(det => `<div>${det.class}: ${(det.confidence * 100).toFixed(1)}%</div>`)
                        .join('');
                }
            });
        }
        
        // Update detections every second
        setInterval(updateDetections, 1000);
    </script>
</body>
</html>

SETUP INSTRUCTIONS:
==================

1. INSTALL DEPENDENCIES:
   pip install flask opencv-python requests numpy python-dotenv

2. CREATE FOLDER STRUCTURE:
   your_project/
   ├── app.py (this file)
   ├── .env (create this file to store your API key)
   └── templates/
       └── index.html (HTML template above)

3. CONFIGURE MODELS:
   - Replace MODEL_ID_1 through MODEL_ID_8 with your actual Roboflow model IDs
   - Add appropriate use case descriptions

4. SET UP .ENV FILE:
   Create a file named '.env' in the project root and add your Roboflow API key:
   ROBOFLOW_API_KEY=your_api_key_here

5. RUN THE APPLICATION:
   python app.py

6. ACCESS THE WEB INTERFACE:
   Open your browser and go to: http://localhost:5000

FEATURES:
=========
- Upload an image for object detection
- Web-based interface with model switching
- Real-time object detection with bounding boxes
- Confidence threshold adjustment
- Toggle detection on/off
- Detection results display
- Responsive design

TROUBLESHOOTING:
===============
- Ensure your Roboflow model IDs are correct and publicly accessible
- Check that your webcam isn't being used by another application
"""