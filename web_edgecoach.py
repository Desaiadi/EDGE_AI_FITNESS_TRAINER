#!/usr/bin/env python3
"""
Web-based EdgeCoach using Flask and browser camera API
This works around opencv compilation issues on ARM64
"""

from flask import Flask, render_template_string, jsonify, request
import json
import time
import threading
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
current_exercise = "squat"
rep_count = 0
start_time = time.time()
fps_counter = 0
fps_start_time = time.time()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeCoach - AI Fitness Coach</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
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
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
            margin: 10px 0;
        }
        .video-container {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        .video-section {
            flex: 1;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        .video-section h3 {
            margin-top: 0;
            color: #ffd700;
        }
        #video {
            width: 100%;
            max-width: 500px;
            height: 375px;
            border-radius: 10px;
            background: #000;
            object-fit: cover;
        }
        .canvas-container {
            position: relative;
            display: inline-block;
        }
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn-primary {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }
        .btn-secondary {
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
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
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #ffd700;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 1.1em;
            opacity: 0.8;
        }
        .feedback {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            min-height: 100px;
            border-left: 5px solid #ffd700;
        }
        .feedback h3 {
            margin-top: 0;
            color: #ffd700;
        }
        .feedback-text {
            font-size: 1.2em;
            line-height: 1.6;
        }
        .status {
            text-align: center;
            margin: 20px 0;
            font-size: 1.3em;
            font-weight: bold;
        }
        .status.ready {
            color: #4ecdc4;
        }
        .status.error {
            color: #ff6b6b;
        }
        .pose-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .landmark {
            position: absolute;
            width: 8px;
            height: 8px;
            background: #ffd700;
            border-radius: 50%;
            border: 2px solid #fff;
            transform: translate(-50%, -50%);
        }
        .connection {
            position: absolute;
            height: 2px;
            background: #ffd700;
            transform-origin: left center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏋️ EdgeCoach</h1>
            <p>AI-Powered Fitness Form Coach</p>
            <p>Real-time pose analysis and form feedback</p>
        </div>

        <div class="video-container">
            <div class="video-section">
                <h3>📹 Camera Feed</h3>
                <div class="canvas-container">
                    <video id="video" autoplay muted playsinline style="display: block;"></video>
                    <canvas id="canvas" width="640" height="480" style="position: absolute; top: 0; left: 0;"></canvas>
                </div>
                <div class="status" id="status">Click "Start Camera" to begin</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="startCamera()">📹 Start Camera</button>
            <button class="btn btn-secondary" onclick="switchExercise('squat')">🏋️ Squat</button>
            <button class="btn btn-secondary" onclick="switchExercise('plank')">🤸 Plank</button>
            <button class="btn btn-success" onclick="resetExercise()">🔄 Reset</button>
            <button class="btn btn-primary" onclick="simulateRep()">💪 Simulate Rep</button>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="exercise-name">SQUAT</div>
                <div class="stat-label">Current Exercise</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="rep-count">0</div>
                <div class="stat-label">Reps Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="timer">0.0s</div>
                <div class="stat-label">Time Elapsed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">0</div>
                <div class="stat-label">FPS</div>
            </div>
        </div>

        <div class="feedback">
            <h3>💬 AI Coach Feedback</h3>
            <div class="feedback-text" id="feedback">
                Welcome to EdgeCoach! Start your camera and begin your workout. 
                I'll analyze your form and provide real-time feedback to help you 
                maintain proper technique and prevent injuries.
            </div>
        </div>
    </div>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let isRunning = false;
        let animationId;
        let frameCount = 0;
        let lastTime = Date.now();

        // Pose landmarks (simplified for demo)
        const POSE_LANDMARKS = {
            NOSE: 0,
            LEFT_EYE: 1,
            RIGHT_EYE: 2,
            LEFT_EAR: 3,
            RIGHT_EAR: 4,
            LEFT_SHOULDER: 5,
            RIGHT_SHOULDER: 6,
            LEFT_ELBOW: 7,
            RIGHT_ELBOW: 8,
            LEFT_WRIST: 9,
            RIGHT_WRIST: 10,
            LEFT_HIP: 11,
            RIGHT_HIP: 12,
            LEFT_KNEE: 13,
            RIGHT_KNEE: 14,
            LEFT_ANKLE: 15,
            RIGHT_ANKLE: 16
        };

        // Pose connections
        const POSE_CONNECTIONS = [
            [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER],
            [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_ELBOW],
            [POSE_LANDMARKS.LEFT_ELBOW, POSE_LANDMARKS.LEFT_WRIST],
            [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_ELBOW],
            [POSE_LANDMARKS.RIGHT_ELBOW, POSE_LANDMARKS.RIGHT_WRIST],
            [POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_HIP],
            [POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_HIP],
            [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP],
            [POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE],
            [POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE],
            [POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE],
            [POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE]
        ];

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: 640, 
                        height: 480,
                        facingMode: 'user'
                    } 
                });
                
                video.srcObject = stream;
                
                // Wait for video to load
                video.onloadedmetadata = function() {
                    video.play();
                    console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
                    
                    document.getElementById('status').textContent = 'Camera started successfully! Video: ' + video.videoWidth + 'x' + video.videoHeight;
                    document.getElementById('status').className = 'status ready';
                    
                    isRunning = true;
                    startPoseDetection();
                };
                
                video.onerror = function(e) {
                    console.error('Video error:', e);
                    document.getElementById('status').textContent = 'Video error: ' + e.message;
                    document.getElementById('status').className = 'status error';
                };
                
            } catch (error) {
                console.error('Error accessing camera:', error);
                document.getElementById('status').textContent = 'Error accessing camera: ' + error.message;
                document.getElementById('status').className = 'status error';
            }
        }

        function startPoseDetection() {
            function detectPose() {
                if (!isRunning) return;
                
                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw video frame from camera
                if (video.videoWidth > 0 && video.videoHeight > 0) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Simulate pose detection (in real app, this would use MediaPipe or similar)
                    simulatePoseDetection();
                }
                
                // Update FPS
                frameCount++;
                const now = Date.now();
                if (now - lastTime >= 1000) {
                    document.getElementById('fps').textContent = frameCount;
                    frameCount = 0;
                    lastTime = now;
                }
                
                animationId = requestAnimationFrame(detectPose);
            }
            
            detectPose();
        }

        function simulatePoseDetection() {
            // Simulate pose landmarks (in real app, these would come from pose detection)
            const landmarks = generateSimulatedLandmarks();
            
            // Draw pose overlay
            drawPoseOverlay(landmarks);
            
            // Analyze form
            analyzeForm(landmarks);
        }

        function generateSimulatedLandmarks() {
            // Generate random but realistic pose landmarks
            const landmarks = [];
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Head
            landmarks.push({x: centerX, y: centerY - 100, confidence: 0.9});
            landmarks.push({x: centerX - 20, y: centerY - 90, confidence: 0.9});
            landmarks.push({x: centerX + 20, y: centerY - 90, confidence: 0.9});
            landmarks.push({x: centerX - 30, y: centerY - 85, confidence: 0.8});
            landmarks.push({x: centerX + 30, y: centerY - 85, confidence: 0.8});
            
            // Shoulders
            landmarks.push({x: centerX - 60, y: centerY - 50, confidence: 0.9});
            landmarks.push({x: centerX + 60, y: centerY - 50, confidence: 0.9});
            
            // Arms
            landmarks.push({x: centerX - 80, y: centerY - 20, confidence: 0.8});
            landmarks.push({x: centerX + 80, y: centerY - 20, confidence: 0.8});
            landmarks.push({x: centerX - 90, y: centerY + 10, confidence: 0.7});
            landmarks.push({x: centerX + 90, y: centerY + 10, confidence: 0.7});
            
            // Hips
            landmarks.push({x: centerX - 40, y: centerY + 20, confidence: 0.9});
            landmarks.push({x: centerX + 40, y: centerY + 20, confidence: 0.9});
            
            // Legs
            landmarks.push({x: centerX - 45, y: centerY + 80, confidence: 0.8});
            landmarks.push({x: centerX + 45, y: centerY + 80, confidence: 0.8});
            landmarks.push({x: centerX - 50, y: centerY + 140, confidence: 0.7});
            landmarks.push({x: centerX + 50, y: centerY + 140, confidence: 0.7});
            
            return landmarks;
        }

        function drawPoseOverlay(landmarks) {
            // Draw landmarks
            landmarks.forEach((landmark, index) => {
                if (landmark.confidence > 0.5) {
                    const x = landmark.x;
                    const y = landmark.y;
                    
                    ctx.fillStyle = '#ffd700';
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 2;
                    
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }
            });
            
            // Draw connections
            ctx.strokeStyle = '#ffd700';
            ctx.lineWidth = 2;
            
            POSE_CONNECTIONS.forEach(([startIdx, endIdx]) => {
                const start = landmarks[startIdx];
                const end = landmarks[endIdx];
                
                if (start && end && start.confidence > 0.5 && end.confidence > 0.5) {
                    ctx.beginPath();
                    ctx.moveTo(start.x, start.y);
                    ctx.lineTo(end.x, end.y);
                    ctx.stroke();
                }
            });
        }

        function analyzeForm(landmarks) {
            const exercise = document.getElementById('exercise-name').textContent.toLowerCase();
            
            if (exercise === 'squat') {
                analyzeSquat(landmarks);
            } else if (exercise === 'plank') {
                analyzePlank(landmarks);
            }
        }

        function analyzeSquat(landmarks) {
            // Simple squat analysis based on hip and knee positions
            const leftHip = landmarks[POSE_LANDMARKS.LEFT_HIP];
            const rightHip = landmarks[POSE_LANDMARKS.RIGHT_HIP];
            const leftKnee = landmarks[POSE_LANDMARKS.LEFT_KNEE];
            const rightKnee = landmarks[POSE_LANDMARKS.RIGHT_KNEE];
            
            if (leftHip && rightHip && leftKnee && rightKnee) {
                const hipHeight = (leftHip.y + rightHip.y) / 2;
                const kneeHeight = (leftKnee.y + rightKnee.y) / 2;
                const depth = hipHeight - kneeHeight;
                
                let feedback = '';
                if (depth > 50) {
                    feedback = '💪 Great depth! Keep your knees over your toes.';
                } else if (depth > 30) {
                    feedback = '⚠️ Try to go deeper - aim for 90 degrees at the knees.';
                } else {
                    feedback = '🏋️ Ready to squat! Lower down slowly.';
                }
                
                document.getElementById('feedback').textContent = feedback;
            }
        }

        function analyzePlank(landmarks) {
            // Simple plank analysis based on body alignment
            const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
            const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
            const leftHip = landmarks[POSE_LANDMARKS.LEFT_HIP];
            const rightHip = landmarks[POSE_LANDMARKS.RIGHT_HIP];
            
            if (leftShoulder && rightShoulder && leftHip && rightHip) {
                const shoulderHeight = (leftShoulder.y + rightShoulder.y) / 2;
                const hipHeight = (leftHip.y + rightHip.y) / 2;
                const alignment = Math.abs(shoulderHeight - hipHeight);
                
                let feedback = '';
                if (alignment < 20) {
                    feedback = '💪 Perfect plank! Keep your body in a straight line.';
                } else if (alignment < 40) {
                    feedback = '⚠️ Slight adjustment needed - straighten your body.';
                } else {
                    feedback = '🏋️ Engage your core and straighten your body.';
                }
                
                document.getElementById('feedback').textContent = feedback;
            }
        }

        function switchExercise(exercise) {
            document.getElementById('exercise-name').textContent = exercise.toUpperCase();
            document.getElementById('feedback').textContent = `Switched to ${exercise} exercise. Get ready to start!`;
            
            // Send to backend
            fetch('/switch_exercise', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({exercise: exercise})
            });
        }

        function resetExercise() {
            document.getElementById('rep-count').textContent = '0';
            document.getElementById('timer').textContent = '0.0s';
            document.getElementById('feedback').textContent = 'Exercise reset! Ready to start fresh.';
            
            // Send to backend
            fetch('/reset_exercise', {method: 'POST'});
        }

        function simulateRep() {
            const currentReps = parseInt(document.getElementById('rep-count').textContent);
            document.getElementById('rep-count').textContent = currentReps + 1;
            
            const exercise = document.getElementById('exercise-name').textContent.toLowerCase();
            let feedback = '';
            
            if (exercise === 'squat') {
                feedback = '💪 Great squat! Keep your form tight.';
            } else if (exercise === 'plank') {
                feedback = '🏋️ Excellent plank hold! Maintain that position.';
            }
            
            document.getElementById('feedback').textContent = feedback;
            
            // Send to backend
            fetch('/simulate_rep', {method: 'POST'});
        }

        // Update timer
        setInterval(() => {
            const startTime = Date.now() - (frameCount * 1000);
            const elapsed = (Date.now() - startTime) / 1000;
            document.getElementById('timer').textContent = elapsed.toFixed(1) + 's';
        }, 100);

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            console.log('EdgeCoach Web App loaded');
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/switch_exercise', methods=['POST'])
def switch_exercise():
    """Switch exercise type"""
    global current_exercise
    data = request.get_json()
    current_exercise = data.get('exercise', 'squat')
    logger.info(f"Switched to {current_exercise} exercise")
    return jsonify({'status': 'success', 'exercise': current_exercise})

@app.route('/reset_exercise', methods=['POST'])
def reset_exercise():
    """Reset exercise"""
    global rep_count, start_time
    rep_count = 0
    start_time = time.time()
    logger.info("Exercise reset")
    return jsonify({'status': 'success'})

@app.route('/simulate_rep', methods=['POST'])
def simulate_rep():
    """Simulate a rep"""
    global rep_count
    rep_count += 1
    logger.info(f"Simulated rep {rep_count}")
    return jsonify({'status': 'success', 'rep_count': rep_count})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get current stats"""
    elapsed_time = time.time() - start_time
    return jsonify({
        'exercise': current_exercise,
        'rep_count': rep_count,
        'elapsed_time': elapsed_time,
        'fps': fps_counter
    })

def main():
    """Main entry point"""
    print("🚀 EdgeCoach Web App - AI Fitness Coach")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
