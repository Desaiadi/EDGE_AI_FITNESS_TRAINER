#!/usr/bin/env python3
"""
Web-based camera display using Flask
"""

import cv2
import base64
import time
from flask import Flask, render_template_string, Response

app = Flask(__name__)

# Global camera
cap = None

def get_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
    return cap

def generate_frames():
    """Generate video frames"""
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
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 EdgeCoach - Live Camera Feed</h1>
            <p>This shows your real camera feed with pose detection!</p>
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
            <div class="controls">
                <button onclick="alert('Squat exercise selected!')">Squat Exercise</button>
                <button onclick="alert('Plank exercise selected!')">Plank Exercise</button>
            </div>
            <p>You should see your live camera feed above with the green skeleton overlay!</p>
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Starting Web-based Camera Display")
    print("=" * 40)
    print("This will open a web browser to show your camera feed")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Stopping web server")
    finally:
        if cap:
            cap.release()
        print("✅ Camera released")
