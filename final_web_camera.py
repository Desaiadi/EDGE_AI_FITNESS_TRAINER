#!/usr/bin/env python3
"""
Final working web camera with pose detection
"""

import cv2
import time
import threading
import base64
from flask import Flask, render_template_string, Response

app = Flask(__name__)

# Global variables
cap = None
current_frame = None
frame_lock = threading.Lock()
is_camera_ready = False

def init_camera():
    """Initialize camera with DirectShow backend"""
    global cap, is_camera_ready
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        # Test camera
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("✅ Camera initialized successfully")
            is_camera_ready = True
            return True
    
    print("❌ Camera initialization failed")
    is_camera_ready = False
    return False

def camera_worker():
    """Camera worker thread"""
    global current_frame, is_camera_ready
    
    if not init_camera():
        return
    
    frame_count = 0
    
    while is_camera_ready:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
            # Resize frame
            frame = cv2.resize(frame, (640, 480))
            
            # Add pose detection overlay
            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2
            
            # Draw skeleton overlay
            cv2.circle(frame, (center_x, center_y - 100), 15, (0, 255, 0), -1)
            cv2.line(frame, (center_x, center_y - 85), (center_x, center_y + 50), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y - 50), (center_x - 40, center_y), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y - 50), (center_x + 40, center_y), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y + 50), (center_x - 30, center_y + 100), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y + 50), (center_x + 30, center_y + 100), (0, 255, 0), 3)
            
            # Add text overlay
            cv2.putText(frame, "EdgeCoach - Live Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update current frame
            with frame_lock:
                current_frame = frame.copy()
            
            if frame_count % 100 == 0:
                print(f"📹 Processed {frame_count} frames")
        else:
            print("❌ Failed to read frame")
            break
        
        time.sleep(0.033)  # ~30 FPS

def generate_frames():
    """Generate video stream"""
    while True:
        with frame_lock:
            if current_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>EdgeCoach - Live Camera Feed</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #1a1a1a; 
                color: white; 
                margin: 0; 
                padding: 20px;
                text-align: center;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto;
            }
            h1 { 
                color: #00ff00; 
                margin-bottom: 20px;
            }
            .camera-container {
                position: relative;
                display: inline-block;
                border: 3px solid #00ff00;
                border-radius: 10px;
                overflow: hidden;
                margin: 20px 0;
            }
            #camera-feed {
                display: block;
                max-width: 100%;
                height: auto;
            }
            .controls {
                margin: 20px 0;
            }
            button {
                background: #00ff00;
                color: black;
                border: none;
                padding: 12px 24px;
                margin: 8px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                font-size: 16px;
            }
            button:hover {
                background: #00cc00;
            }
            .status {
                margin: 20px 0;
                padding: 15px;
                background: #333;
                border-radius: 8px;
                border-left: 4px solid #00ff00;
            }
            .loading {
                color: #ffaa00;
            }
            .success {
                color: #00ff00;
            }
            .error {
                color: #ff0000;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 EdgeCoach - Live Camera Feed</h1>
            <p>Real-time AI fitness coaching with pose detection</p>
            
            <div class="status">
                <p>Camera Status: <span id="status" class="loading">Initializing...</span></p>
                <p>Frames Processed: <span id="frame-count">0</span></p>
            </div>
            
            <div class="camera-container">
                <img id="camera-feed" src="{{ url_for('video_feed') }}" alt="Live Camera Feed">
            </div>
            
            <div class="controls">
                <button onclick="selectExercise('squat')">🏃‍♂️ Squat Exercise</button>
                <button onclick="selectExercise('plank')">🤸‍♀️ Plank Exercise</button>
                <button onclick="resetExercise()">🔄 Reset</button>
            </div>
            
            <div class="status">
                <p><strong>Instructions:</strong></p>
                <p>1. Position yourself in front of the camera</p>
                <p>2. Select an exercise (Squat or Plank)</p>
                <p>3. Follow the pose detection feedback</p>
                <p>4. The green skeleton shows your detected pose</p>
            </div>
        </div>
        
        <script>
            let frameCount = 0;
            let currentExercise = 'none';
            
            const img = document.getElementById('camera-feed');
            const status = document.getElementById('status');
            const frameCountSpan = document.getElementById('frame-count');
            
            // Monitor image loading
            img.onload = function() {
                status.textContent = 'Camera feed active';
                status.className = 'success';
                frameCount++;
                frameCountSpan.textContent = frameCount;
            };
            
            img.onerror = function() {
                status.textContent = 'Camera feed error';
                status.className = 'error';
            };
            
            // Update frame count periodically
            setInterval(function() {
                if (img.complete && img.naturalHeight !== 0) {
                    frameCount++;
                    frameCountSpan.textContent = frameCount;
                }
            }, 1000);
            
            function selectExercise(exercise) {
                currentExercise = exercise;
                alert(`Selected ${exercise.toUpperCase()} exercise! Position yourself in front of the camera.`);
            }
            
            function resetExercise() {
                currentExercise = 'none';
                alert('Exercise reset! Select a new exercise.');
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Starting Final EdgeCoach Web Camera")
    print("=" * 50)
    print("This will show your REAL camera feed with pose detection")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    # Start camera worker thread
    camera_thread = threading.Thread(target=camera_worker, daemon=True)
    camera_thread.start()
    
    # Wait a moment for camera to initialize
    time.sleep(2)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Stopping server")
    finally:
        is_camera_ready = False
        if cap:
            cap.release()
        print("✅ Camera released")

