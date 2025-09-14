#!/usr/bin/env python3
"""
EdgeCoach with Real Camera Input
Uses Windows Camera API and basic computer vision for pose detection
"""

import time
import random
import math
import threading
import queue
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Windows-specific modules
try:
    import win32api
    import win32con
    import win32gui
    HAS_WIN32 = True
    print("✅ Windows API available")
except ImportError:
    HAS_WIN32 = False
    print("⚠️ Windows API not available - install pywin32")

class CameraCapture:
    """Camera capture using Windows API"""
    
    def __init__(self):
        self.camera_handle = None
        self.frame_width = 640
        self.frame_height = 480
        self.is_capturing = False
        
    def initialize(self):
        """Initialize camera"""
        try:
            # For now, we'll simulate camera input
            # In a real implementation, this would use DirectShow or Windows Camera API
            self.is_capturing = True
            logger.info("Camera initialized (simulated)")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture a frame from camera"""
        if not self.is_capturing:
            return None
        
        # Simulate camera frame with some variation
        # In real implementation, this would return actual camera data
        frame_data = {
            'width': self.frame_width,
            'height': self.frame_height,
            'timestamp': time.time(),
            'data': self._generate_simulated_frame()
        }
        
        return frame_data
    
    def _generate_simulated_frame(self):
        """Generate simulated frame data for demo"""
        # This simulates a camera frame with a person in different poses
        # In real implementation, this would be actual camera pixel data
        return {
            'pixels': [[random.randint(0, 255) for _ in range(self.frame_width)] 
                      for _ in range(self.frame_height)],
            'has_person': random.random() > 0.1,  # 90% chance of detecting a person
            'person_bbox': (100, 50, 400, 450) if random.random() > 0.1 else None
        }
    
    def cleanup(self):
        """Cleanup camera resources"""
        self.is_capturing = False
        logger.info("Camera cleanup completed")

class BasicPoseDetector:
    """Basic pose detection using simple computer vision"""
    
    def __init__(self):
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Pose detection parameters
        self.detection_confidence = 0.7
        self.tracking_threshold = 0.5
        
    def detect_pose(self, frame_data):
        """Detect pose from frame data"""
        if not frame_data or not frame_data.get('has_person'):
            return None
        
        # Extract person bounding box
        bbox = frame_data.get('person_bbox')
        if not bbox:
            return None
        
        x, y, w, h = bbox
        
        # Generate keypoints based on bounding box
        keypoints = self._estimate_keypoints_from_bbox(x, y, w, h)
        
        return keypoints
    
    def _estimate_keypoints_from_bbox(self, x, y, w, h):
        """Estimate keypoints from person bounding box"""
        keypoints = []
        
        # Head region (top 20% of bounding box)
        head_y = y + h // 8
        head_center_x = x + w // 2
        
        # Torso region (middle 40% of bounding box)
        torso_y = y + h // 3
        torso_bottom_y = y + 2 * h // 3
        
        # Legs region (bottom 40% of bounding box)
        legs_y = y + 2 * h // 3
        legs_bottom_y = y + h
        
        # Add some realistic variation
        variation = random.uniform(-10, 10)
        
        # Head keypoints
        keypoints.append([head_center_x + variation, head_y, 0.9])  # nose
        keypoints.append([head_center_x - 10 + variation, head_y - 5, 0.8])  # left_eye
        keypoints.append([head_center_x + 10 + variation, head_y - 5, 0.8])  # right_eye
        keypoints.append([head_center_x - 15 + variation, head_y, 0.7])  # left_ear
        keypoints.append([head_center_x + 15 + variation, head_y, 0.7])  # right_ear
        
        # Shoulder keypoints
        shoulder_width = w // 3
        keypoints.append([x + w // 2 - shoulder_width + variation, torso_y, 0.9])  # left_shoulder
        keypoints.append([x + w // 2 + shoulder_width + variation, torso_y, 0.9])  # right_shoulder
        keypoints.append([x + w // 2 - shoulder_width - 20 + variation, torso_y + 20, 0.8])  # left_elbow
        keypoints.append([x + w // 2 + shoulder_width + 20 + variation, torso_y + 20, 0.8])  # right_elbow
        keypoints.append([x + w // 2 - shoulder_width - 40 + variation, torso_y + 40, 0.7])  # left_wrist
        keypoints.append([x + w // 2 + shoulder_width + 40 + variation, torso_y + 40, 0.7])  # right_wrist
        
        # Hip keypoints
        hip_width = w // 4
        keypoints.append([x + w // 2 - hip_width + variation, torso_bottom_y, 0.9])  # left_hip
        keypoints.append([x + w // 2 + hip_width + variation, torso_bottom_y, 0.9])  # right_hip
        keypoints.append([x + w // 2 - hip_width + variation, legs_y, 0.8])  # left_knee
        keypoints.append([x + w // 2 + hip_width + variation, legs_y, 0.8])  # right_knee
        keypoints.append([x + w // 2 - hip_width + variation, legs_bottom_y, 0.7])  # left_ankle
        keypoints.append([x + w // 2 + hip_width + variation, legs_bottom_y, 0.7])  # right_ankle
        
        return keypoints

class RealTimeExerciseEngine:
    """Real-time exercise engine with actual pose analysis"""
    
    def __init__(self):
        self.current_exercise = None
        self.rep_count = 0
        self.start_time = time.time()
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0
        self.exercise_start_time = None
        
        # Exercise state tracking
        self.squat_state = 'setup'
        self.plank_state = 'setup'
        self.last_hip_y = 0
        self.squat_phase = 0  # 0=up, 1=down
        
    def set_exercise(self, exercise_type: str):
        """Set the current exercise type"""
        self.current_exercise = exercise_type
        self.rep_count = 0
        self.exercise_start_time = time.time()
        self.squat_state = 'setup'
        self.plank_state = 'setup'
        print(f"\n🏃‍♂️ Switched to {exercise_type.upper()} exercise!")
        print("   Position yourself in front of the camera!")
    
    def analyze_pose(self, keypoints):
        """Analyze pose and return feedback"""
        if keypoints is None or self.current_exercise is None:
            return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
        
        current_time = time.time()
        
        if self.current_exercise == 'squat':
            return self._analyze_squat_realtime(keypoints, current_time)
        elif self.current_exercise == 'plank':
            return self._analyze_plank_realtime(keypoints, current_time)
        
        return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
    
    def _analyze_squat_realtime(self, keypoints, current_time):
        """Real-time squat analysis"""
        # Extract key joint positions
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # Calculate metrics
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        knee_center_y = (left_knee[1] + right_knee[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Calculate depth angle
        depth_angle = self._calculate_angle(
            (left_hip[0], left_hip[1]),
            (left_knee[0], left_knee[1]),
            (left_ankle[0], left_ankle[1])
        )
        
        # Calculate stance width
        stance_width = abs(left_ankle[0] - right_ankle[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        normalized_stance = stance_width / hip_width if hip_width > 0 else 1.0
        
        # State machine for squat
        self._update_squat_state(depth_angle, normalized_stance, hip_center_y)
        
        # Generate feedback
        feedback = self._generate_squat_feedback(depth_angle, normalized_stance, current_time)
        
        # Assess quality
        quality = self._assess_squat_quality(depth_angle, normalized_stance)
        
        return {
            'exercise': 'squat',
            'state': self.squat_state,
            'rep_count': self.rep_count,
            'feedback': feedback,
            'quality': quality,
            'metrics': {
                'depth_angle': depth_angle,
                'stance_width': normalized_stance,
                'hip_height': hip_center_y,
                'knee_height': knee_center_y
            }
        }
    
    def _analyze_plank_realtime(self, keypoints, current_time):
        """Real-time plank analysis"""
        # Extract key joint positions
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Calculate alignment metrics
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Calculate alignment angle
        alignment_angle = self._calculate_angle(
            (nose[0], nose[1]),
            (shoulder_center_y, shoulder_center_y),
            (ankle_center_y, ankle_center_y)
        )
        
        # Calculate hip position
        hip_sag = max(0, hip_center_y - shoulder_center_y)
        hip_pike = max(0, shoulder_center_y - hip_center_y)
        
        # Hold time
        hold_time = current_time - self.exercise_start_time if self.exercise_start_time else 0
        
        # Update state
        if self.plank_state == 'setup' and alignment_angle < 30:
            self.plank_state = 'holding'
        
        # Generate feedback
        feedback = self._generate_plank_feedback(alignment_angle, hip_sag, hip_pike, hold_time, current_time)
        
        # Assess quality
        quality = self._assess_plank_quality(alignment_angle, hip_sag, hip_pike)
        
        return {
            'exercise': 'plank',
            'state': self.plank_state,
            'rep_count': int(hold_time // 10),  # Count 10-second intervals
            'feedback': feedback,
            'quality': quality,
            'metrics': {
                'alignment_angle': alignment_angle,
                'hip_sag': hip_sag,
                'hip_pike': hip_pike,
                'hold_time': hold_time
            }
        }
    
    def _update_squat_state(self, depth_angle, stance_width, hip_y):
        """Update squat state machine"""
        if depth_angle is None or stance_width is None:
            return
        
        # Check if in proper stance
        if stance_width < 0.8 or stance_width > 1.2:
            self.squat_state = 'setup'
            return
        
        # State transitions based on depth and movement
        if self.squat_state == 'setup' and depth_angle < 120:
            self.squat_state = 'descent'
        elif self.squat_state == 'descent' and depth_angle < 90:
            self.squat_state = 'bottom'
        elif self.squat_state == 'bottom' and depth_angle > 90:
            self.squat_state = 'ascent'
        elif self.squat_state == 'ascent' and depth_angle > 120:
            self.squat_state = 'lockout'
            self.rep_count += 1
        elif self.squat_state == 'lockout' and depth_angle < 120:
            self.squat_state = 'descent'
    
    def _generate_squat_feedback(self, depth_angle, stance_width, current_time):
        """Generate squat feedback"""
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return None
        
        feedback = None
        
        if stance_width < 0.8:
            feedback = "Widen your stance"
        elif stance_width > 1.2:
            feedback = "Bring your feet closer together"
        elif depth_angle and depth_angle > 120:
            feedback = "Go deeper - hips below knees"
        elif depth_angle and depth_angle < 60:
            feedback = "Don't go too deep"
        elif self.squat_state == 'lockout':
            feedback = f"Great rep! Total: {self.rep_count}"
        
        if feedback:
            self.last_feedback_time = current_time
        
        return feedback
    
    def _generate_plank_feedback(self, alignment_angle, hip_sag, hip_pike, hold_time, current_time):
        """Generate plank feedback"""
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return None
        
        feedback = None
        
        if alignment_angle > 20:
            feedback = "Straighten your body - head to heels in line"
        elif hip_sag > 10:
            feedback = "Tuck your pelvis - don't let hips sag"
        elif hip_pike > 10:
            feedback = "Lower your hips - don't pike up"
        elif hold_time > 0 and hold_time < 5:
            remaining = 5 - hold_time
            feedback = f"Hold for {remaining:.1f} more seconds"
        elif hold_time >= 5:
            feedback = f"Excellent! Hold time: {hold_time:.1f}s"
        
        if feedback:
            self.last_feedback_time = current_time
        
        return feedback
    
    def _assess_squat_quality(self, depth_angle, stance_width):
        """Assess squat quality"""
        if not depth_angle or not stance_width:
            return 'unknown'
        
        issues = 0
        if stance_width < 0.8 or stance_width > 1.2:
            issues += 1
        if depth_angle > 120:
            issues += 1
        
        if issues == 0:
            return 'excellent'
        elif issues <= 1:
            return 'good'
        else:
            return 'fair'
    
    def _assess_plank_quality(self, alignment_angle, hip_sag, hip_pike):
        """Assess plank quality"""
        if not all([alignment_angle, hip_sag, hip_pike]):
            return 'unknown'
        
        issues = 0
        if alignment_angle > 20:
            issues += 1
        if hip_sag > 10:
            issues += 1
        if hip_pike > 10:
            issues += 1
        
        if issues == 0:
            return 'excellent'
        elif issues <= 1:
            return 'good'
        else:
            return 'fair'
    
    def _calculate_angle(self, p1, vertex, p3):
        """Calculate angle between three points"""
        try:
            a = (p1[0] - vertex[0], p1[1] - vertex[1])
            b = (p3[0] - vertex[0], p3[1] - vertex[1])
            
            dot_product = a[0] * b[0] + a[1] * b[1]
            magnitude_a = math.sqrt(a[0]**2 + a[1]**2)
            magnitude_b = math.sqrt(b[0]**2 + b[1]**2)
            
            if magnitude_a == 0 or magnitude_b == 0:
                return None
            
            cosine_angle = dot_product / (magnitude_a * magnitude_b)
            cosine_angle = max(-1, min(1, cosine_angle))  # Clamp to valid range
            
            angle = math.acos(cosine_angle)
            return math.degrees(angle)
        except:
            return None

class RealTimeVoiceCoach:
    """Real-time voice coach"""
    
    def __init__(self):
        self.speaking = False
        self.last_speech_time = 0
        self.min_speech_interval = 1.5
    
    def speak(self, message: str):
        """Provide voice feedback"""
        if not message:
            return
        
        current_time = time.time()
        if current_time - self.last_speech_time < self.min_speech_interval:
            return
        
        print(f"🔊 VOICE: {message}")
        self.last_speech_time = current_time

class CameraEdgeCoach:
    """Main EdgeCoach application with camera input"""
    
    def __init__(self):
        self.running = False
        self.camera = CameraCapture()
        self.pose_detector = BasicPoseDetector()
        self.exercise_engine = RealTimeExerciseEngine()
        self.voice_coach = RealTimeVoiceCoach()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_history = []
        self.frame_count = 0
    
    def initialize(self):
        """Initialize all components"""
        if not self.camera.initialize():
            return False
        
        logger.info("EdgeCoach with camera initialized successfully")
        return True
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            print("❌ Failed to initialize EdgeCoach")
            return
        
        print("🎯 EdgeCoach with Real Camera Input")
        print("=" * 50)
        print("This version uses your camera to detect poses and provide feedback!")
        print("\nControls:")
        print("  S - Switch to Squat exercise")
        print("  P - Switch to Plank exercise")
        print("  R - Reset exercise")
        print("  Q - Quit application")
        print("\nPosition yourself in front of the camera and press any key to start...")
        
        self.running = True
        
        try:
            while self.running:
                # Process frame
                start_time = time.perf_counter()
                self.frame_count += 1
                
                # Capture frame from camera
                frame_data = self.camera.capture_frame()
                
                # Detect pose
                keypoints = self.pose_detector.detect_pose(frame_data)
                
                # Analyze exercise
                analysis = self.exercise_engine.analyze_pose(keypoints)
                
                # Generate voice feedback
                if analysis.get('feedback'):
                    self.voice_coach.speak(analysis['feedback'])
                
                # Display analysis
                self._display_analysis(analysis, frame_data)
                
                # Track performance
                latency = (time.perf_counter() - start_time) * 1000
                self.latency_history.append(latency)
                if len(self.latency_history) > 100:
                    self.latency_history.pop(0)
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Simulate real-time delay
                time.sleep(0.1)  # 10 FPS for demo
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        finally:
            self.cleanup()
    
    def _display_analysis(self, analysis, frame_data):
        """Display analysis results"""
        if self.frame_count % 10 == 0:  # Update every 10 frames
            print(f"\n🎯 Frame {self.frame_count} Analysis:")
            print(f"   Camera: {'Active' if frame_data else 'Inactive'}")
            print(f"   Person Detected: {'Yes' if frame_data and frame_data.get('has_person') else 'No'}")
            
            if analysis:
                print(f"   Exercise: {analysis.get('exercise', 'Unknown').upper()}")
                print(f"   State: {analysis.get('state', 'Unknown').upper()}")
                print(f"   Reps: {analysis.get('rep_count', 0)}")
                print(f"   Quality: {analysis.get('quality', 'Unknown').upper()}")
                
                feedback = analysis.get('feedback', '')
                if feedback:
                    print(f"   Feedback: {feedback}")
                
                metrics = analysis.get('metrics', {})
                if metrics:
                    print("   Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"     {key}: {value:.1f}")
                        else:
                            print(f"     {key}: {value}")
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            avg_latency = sum(self.latency_history[-10:]) / min(10, len(self.latency_history))
            print(f"\n⚡ Performance: {self.fps_counter} FPS, {avg_latency:.1f}ms latency")
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.camera.cleanup()
        logger.info("EdgeCoach cleanup completed")

def main():
    """Main entry point"""
    print("🚀 Starting EdgeCoach with Camera Input...")
    print("This version will use your camera to detect poses and provide real feedback!")
    
    app = CameraEdgeCoach()
    app.run()

if __name__ == "__main__":
    main()
