#!/usr/bin/env python3
"""
Simple Web Camera - Basic camera feed without complex overlays
"""

from flask import Flask, render_template_string
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Simple HTML template focused on camera display
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeCoach - Camera Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        #video {
            width: 100%;
            max-width: 640px;
            height: 480px;
            border-radius: 10px;
            background: #000;
            border: 3px solid #ffd700;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 0 10px;
        }
        .btn-primary {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }
        .btn-success {
            background: linear-gradient(45deg, #45b7d1, #96c93d);
            color: white;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        .status {
            text-align: center;
            margin: 20px 0;
            font-size: 1.2em;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.3);
        }
        .status.ready {
            color: #4ecdc4;
            border: 2px solid #4ecdc4;
        }
        .status.error {
            color: #ff6b6b;
            border: 2px solid #ff6b6b;
        }
        .status.waiting {
            color: #ffd700;
            border: 2px solid #ffd700;
        }
        .info {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .info h3 {
            margin-top: 0;
            color: #ffd700;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 EdgeCoach Camera Test</h1>
            <p>Testing camera access and video feed display</p>
        </div>

        <div class="video-container">
            <video id="video" autoplay muted playsinline></video>
        </div>

        <div class="status waiting" id="status">
            Click "Start Camera" to begin testing
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="startCamera()">📹 Start Camera</button>
            <button class="btn btn-success" onclick="stopCamera()">⏹️ Stop Camera</button>
        </div>

        <div class="info">
            <h3>🔧 Debug Information</h3>
            <p><strong>Camera Status:</strong> <span id="camera-status">Not started</span></p>
            <p><strong>Video Dimensions:</strong> <span id="video-dims">N/A</span></p>
            <p><strong>Stream Active:</strong> <span id="stream-status">No</span></p>
            <p><strong>Browser:</strong> <span id="browser-info">Loading...</span></p>
        </div>

        <div class="info">
            <h3>📋 Troubleshooting</h3>
            <ul>
                <li>Make sure you allow camera permissions when prompted</li>
                <li>Check that no other app is using your camera</li>
                <li>Try refreshing the page if camera doesn't start</li>
                <li>Check browser console (F12) for any error messages</li>
            </ul>
        </div>
    </div>

    <script>
        let video = document.getElementById('video');
        let stream = null;

        // Detect browser
        document.getElementById('browser-info').textContent = navigator.userAgent;

        async function startCamera() {
            try {
                console.log('Starting camera...');
                document.getElementById('status').textContent = 'Requesting camera access...';
                document.getElementById('status').className = 'status waiting';
                
                // Request camera access
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    } 
                });
                
                console.log('Camera stream obtained:', stream);
                
                // Set video source
                video.srcObject = stream;
                
                // Wait for video to load
                video.onloadedmetadata = function() {
                    console.log('Video metadata loaded');
                    console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
                    
                    // Update status
                    document.getElementById('status').textContent = 'Camera started successfully!';
                    document.getElementById('status').className = 'status ready';
                    
                    // Update debug info
                    document.getElementById('camera-status').textContent = 'Active';
                    document.getElementById('video-dims').textContent = video.videoWidth + ' x ' + video.videoHeight;
                    document.getElementById('stream-status').textContent = 'Yes';
                    
                    console.log('Camera setup complete');
                };
                
                video.onerror = function(e) {
                    console.error('Video error:', e);
                    document.getElementById('status').textContent = 'Video error: ' + e.message;
                    document.getElementById('status').className = 'status error';
                };
                
                // Play video
                video.play().then(() => {
                    console.log('Video playing');
                }).catch(e => {
                    console.error('Video play error:', e);
                    document.getElementById('status').textContent = 'Video play error: ' + e.message;
                    document.getElementById('status').className = 'status error';
                });
                
            } catch (error) {
                console.error('Camera error:', error);
                document.getElementById('status').textContent = 'Camera error: ' + error.message;
                document.getElementById('status').className = 'status error';
                document.getElementById('camera-status').textContent = 'Error: ' + error.name;
            }
        }

        function stopCamera() {
            if (stream) {
                console.log('Stopping camera...');
                stream.getTracks().forEach(track => {
                    track.stop();
                    console.log('Track stopped:', track.kind);
                });
                stream = null;
                video.srcObject = null;
                
                document.getElementById('status').textContent = 'Camera stopped';
                document.getElementById('status').className = 'status waiting';
                document.getElementById('camera-status').textContent = 'Stopped';
                document.getElementById('video-dims').textContent = 'N/A';
                document.getElementById('stream-status').textContent = 'No';
            }
        }

        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            document.getElementById('status').textContent = 'Camera not supported in this browser';
            document.getElementById('status').className = 'status error';
        }

        // Log when page loads
        console.log('EdgeCoach Camera Test loaded');
        console.log('getUserMedia supported:', !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

def main():
    """Main entry point"""
    print("🎥 EdgeCoach Camera Test - Simple Version")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == "__main__":
    main()
