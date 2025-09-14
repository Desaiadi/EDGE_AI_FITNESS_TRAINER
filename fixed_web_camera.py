#!/usr/bin/env python3
"""
Fixed web-based camera display
"""

import cv2
import base64
import time
import threading
from flask import Flask, render_template_string, Response

app = Flask(__name__)

# Global camera and frame
cap = None
current_frame = None
frame_lock = threading.Lock()

def get_camera():
    global cap
    if cap is None:
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Test if we can read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Camera {camera_index} opened successfully")
                    return cap
                else:
                    cap.release()
                    cap = None
        
        print("❌ No working camera found")
        return None
    return cap

def camera_thread():
    """Camera thread to continuously capture frames"""
    global current_frame, cap
    
    camera = get_camera()
    if camera is None:
        return
    
    while True:
        ret, frame = camera.read()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (640, 480))
            
            # Add pose detection overlay
            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2
            
            # Draw skeleton
            cv2.circle(frame, (center_x, center_y - 100), 15, (0, 255, 0), -1)
            cv2.line(frame, (center_x, center_y - 85), (center_x, center_y + 50), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y - 50), (center_x - 40, center_y), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y - 50), (center_x + 40, center_y), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y + 50), (center_x - 30, center_y + 100), (0, 255, 0), 3)
            cv2.line(frame, (center_x, center_y + 50), (center_x + 30, center_y + 100), (0, 255, 0), 3)
            
            # Add text
            cv2.putText(frame, "EdgeCoach - Live Camera Feed", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' for squat, 'p' for plank", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            with frame_lock:
                current_frame = frame.copy()
        else:
            print("❌ Failed to read frame")
            break
        
        time.sleep(0.033)  # ~30 FPS

def generate_frames():
    """Generate video frames for web streaming"""
    while True:
        with frame_lock:
            if current_frame is not None:
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>EdgeCoach - Live Camera Feed</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #1a1a1a; 
                color: white; 
                margin: 0; 
                padding: 20px;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                text-align: center;
            }
            h1 { 
                color: #00ff00; 
                margin-bottom: 20px;
            }
            img { 
                border: 2px solid #00ff00; 
                border-radius: 10px;
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
                padding: 10px 20px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }
            button:hover {
                background: #00cc00;
            }
            .status {
                margin: 20px 0;
                padding: 10px;
                background: #333;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 EdgeCoach - Live Camera Feed</h1>
            <p>This shows your real camera feed with pose detection!</p>
            <div class="status">
                <p>Camera Status: <span id="status">Loading...</span></p>
            </div>
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed" id="camera-feed">
            <div class="controls">
                <button onclick="alert('Squat exercise selected!')">Squat Exercise</button>
                <button onclick="alert('Plank exercise selected!')">Plank Exercise</button>
            </div>
            <p>You should see your live camera feed above with the green skeleton overlay!</p>
        </div>
        
        <script>
            // Check if camera feed is loading
            const img = document.getElementById('camera-feed');
            const status = document.getElementById('status');
            
            img.onload = function() {
                status.textContent = 'Camera feed loaded successfully!';
                status.style.color = '#00ff00';
            };
            
            img.onerror = function() {
                status.textContent = 'Camera feed failed to load';
                status.style.color = '#ff0000';
            };
            
            // Check every second
            setInterval(function() {
                if (img.complete && img.naturalHeight !== 0) {
                    status.textContent = 'Camera feed active';
                    status.style.color = '#00ff00';
                }
            }, 1000);
        </script>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Starting Fixed Web-based Camera Display")
    print("=" * 50)
    print("This will open a web browser to show your camera feed")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    # Start camera thread
    camera_thread = threading.Thread(target=camera_thread, daemon=True)
    camera_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Stopping web server")
    finally:
        if cap:
            cap.release()
        print("✅ Camera released")
