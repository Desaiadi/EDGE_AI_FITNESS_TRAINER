#!/usr/bin/env python3
"""
EdgeCoach with Visual Camera Feed and Pose Recognition
Shows real-time camera feed with pose detection overlay
"""

import time
import random
import math
import os
import sys
import threading
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OpenCV for visual display
try:
    import cv2
    HAS_OPENCV = True
    print("✅ OpenCV available - will show visual feed")
except ImportError:
    HAS_OPENCV = False
    print("⚠️ OpenCV not available - will use console display")

class VisualPoseDetector:
    """Visual pose detector with real-time display"""
    
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
        
        # Camera setup
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        if HAS_OPENCV:
            self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera for visual display"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print("✅ Camera initialized successfully")
            else:
                print("⚠️ Camera not accessible - using simulation")
                self.cap = None
        except Exception as e:
            print(f"⚠️ Camera error: {e} - using simulation")
            self.cap = None
    
    def detect_pose_from_camera(self):
        """Detect pose from camera feed"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Detect pose in the frame
                keypoints = self._detect_pose_in_frame(frame)
                return keypoints, frame
        
        # Fallback to simulation
        return self._simulate_pose_detection(), None
    
    def _detect_pose_in_frame(self, frame):
        """Detect pose in camera frame using basic computer vision"""
        height, width = frame.shape[:2]
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Estimate pose based on contours
        keypoints = self._estimate_keypoints_from_contours(contours, width, height)
        
        return keypoints
    
    def _estimate_keypoints_from_contours(self, contours, width, height):
        """Estimate keypoints from detected contours"""
        if not contours:
            return self._get_default_pose(width, height)
        
        # Find the largest contour (likely the person)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Estimate keypoints based on bounding box proportions
        center_x = x + w // 2
        head_y = y + h // 8
        shoulder_y = y + h // 4
        hip_y = y + h // 2
        knee_y = y + 3 * h // 4
        ankle_y = y + 7 * h // 8
        
        # Generate keypoints with some variation
        variation = random.uniform(-10, 10)
        
        keypoints = [
            # Head
            [center_x + variation, head_y, 0.9],
            [center_x - 10 + variation, head_y - 5, 0.8],
            [center_x + 10 + variation, head_y - 5, 0.8],
            [center_x - 15 + variation, head_y, 0.7],
            [center_x + 15 + variation, head_y, 0.7],
            
            # Shoulders
            [center_x - 30 + variation, shoulder_y, 0.9],
            [center_x + 30 + variation, shoulder_y, 0.9],
            [center_x - 50 + variation, shoulder_y + 20, 0.8],
            [center_x + 50 + variation, shoulder_y + 20, 0.8],
            [center_x - 70 + variation, shoulder_y + 40, 0.7],
            [center_x + 70 + variation, shoulder_y + 40, 0.7],
            
            # Hips
            [center_x - 25 + variation, hip_y, 0.9],
            [center_x + 25 + variation, hip_y, 0.9],
            [center_x - 30 + variation, knee_y, 0.8],
            [center_x + 30 + variation, knee_y, 0.8],
            [center_x - 25 + variation, ankle_y, 0.7],
            [center_x + 25 + variation, ankle_y, 0.7],
        ]
        
        return keypoints
    
    def _get_default_pose(self, width, height):
        """Get default pose when no person is detected"""
        center_x = width // 2
        center_y = height // 2
        
        keypoints = []
        # Generate a standing pose
        keypoints.append([center_x, center_y - 100, 0.5])  # nose
        keypoints.append([center_x - 10, center_y - 110, 0.4])  # left_eye
        keypoints.append([center_x + 10, center_y - 110, 0.4])  # right_eye
        keypoints.append([center_x - 20, center_y - 105, 0.3])  # left_ear
        keypoints.append([center_x + 20, center_y - 105, 0.3])  # right_ear
        keypoints.append([center_x - 40, center_y - 50, 0.6])  # left_shoulder
        keypoints.append([center_x + 40, center_y - 50, 0.6])  # right_shoulder
        keypoints.append([center_x - 60, center_y - 20, 0.5])  # left_elbow
        keypoints.append([center_x + 60, center_y - 20, 0.5])  # right_elbow
        keypoints.append([center_x - 80, center_y + 10, 0.4])  # left_wrist
        keypoints.append([center_x + 80, center_y + 10, 0.4])  # right_wrist
        keypoints.append([center_x - 30, center_y + 50, 0.7])  # left_hip
        keypoints.append([center_x + 30, center_y + 50, 0.7])  # right_hip
        keypoints.append([center_x - 35, center_y + 100, 0.6])  # left_knee
        keypoints.append([center_x + 35, center_y + 100, 0.6])  # right_knee
        keypoints.append([center_x - 30, center_y + 150, 0.5])  # left_ankle
        keypoints.append([center_x + 30, center_y + 150, 0.5])  # right_ankle
        
        return keypoints
    
    def _simulate_pose_detection(self):
        """Simulate pose detection for demo"""
        # Generate realistic pose data with movement
        keypoints = []
        center_x = 320
        center_y = 240
        
        # Add some variation to simulate real movement
        variation = random.uniform(-20, 20)
        
        keypoints.append([center_x + variation, center_y - 100, 0.9])  # nose
        keypoints.append([center_x - 10 + variation, center_y - 110, 0.8])  # left_eye
        keypoints.append([center_x + 10 + variation, center_y - 110, 0.8])  # right_eye
        keypoints.append([center_x - 20 + variation, center_y - 105, 0.7])  # left_ear
        keypoints.append([center_x + 20 + variation, center_y - 105, 0.7])  # right_ear
        keypoints.append([center_x - 40 + variation, center_y - 50, 0.9])  # left_shoulder
        keypoints.append([center_x + 40 + variation, center_y - 50, 0.9])  # right_shoulder
        keypoints.append([center_x - 60 + variation, center_y - 20, 0.8])  # left_elbow
        keypoints.append([center_x + 60 + variation, center_y - 20, 0.8])  # right_elbow
        keypoints.append([center_x - 80 + variation, center_y + 10, 0.7])  # left_wrist
        keypoints.append([center_x + 80 + variation, center_y + 10, 0.7])  # right_wrist
        keypoints.append([center_x - 30 + variation, center_y + 50, 0.9])  # left_hip
        keypoints.append([center_x + 30 + variation, center_y + 50, 0.9])  # right_hip
        keypoints.append([center_x - 35 + variation, center_y + 100, 0.8])  # left_knee
        keypoints.append([center_x + 35 + variation, center_y + 100, 0.8])  # right_knee
        keypoints.append([center_x - 30 + variation, center_y + 150, 0.7])  # left_ankle
        keypoints.append([center_x + 30 + variation, center_y + 150, 0.7])  # right_ankle
        
        return keypoints
    
    def draw_skeleton(self, frame, keypoints, confidence_threshold=0.3):
        """Draw skeleton overlay on frame"""
        if frame is None or keypoints is None:
            return frame
        
        # Define skeleton connections
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > confidence_threshold:
                color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                cv2.putText(frame, str(i), (int(x), int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw skeleton connections
        for start_idx, end_idx in skeleton_connections:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            if (start_point[2] > confidence_threshold and 
                end_point[2] > confidence_threshold):
                
                start_pos = (int(start_point[0]), int(start_point[1]))
                end_pos = (int(end_point[0]), int(end_point[1]))
                cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)
        
        return frame
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        if HAS_OPENCV:
            cv2.destroyAllWindows()

class VisualExerciseEngine:
    """Visual exercise engine with real-time analysis"""
    
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

class VisualEdgeCoach:
    """Main EdgeCoach application with visual display"""
    
    def __init__(self):
        self.running = False
        self.pose_detector = VisualPoseDetector()
        self.exercise_engine = VisualExerciseEngine()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_history = []
        self.frame_count = 0
    
    def run(self):
        """Main application loop with visual display"""
        print("🎯 EdgeCoach with Visual Camera Feed")
        print("=" * 50)
        print("This version shows real-time camera feed with pose detection!")
        
        if not HAS_OPENCV:
            print("⚠️ OpenCV not available - install opencv-python for visual display")
            print("   Running in console mode...")
            self._run_console_mode()
            return
        
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
                
                # Get pose detection from camera
                keypoints, frame = self.pose_detector.detect_pose_from_camera()
                
                # Analyze exercise
                analysis = self.exercise_engine.analyze_pose(keypoints)
                
                # Draw visual overlay
                if frame is not None:
                    # Draw skeleton
                    frame = self.pose_detector.draw_skeleton(frame, keypoints)
                    
                    # Draw UI overlay
                    frame = self._draw_ui_overlay(frame, analysis)
                    
                    # Display frame
                    cv2.imshow('EdgeCoach - Real-time AI Fitness Coach', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.exercise_engine.set_exercise('squat')
                    elif key == ord('p'):
                        self.exercise_engine.set_exercise('plank')
                    elif key == ord('r'):
                        self.exercise_engine.rep_count = 0
                        self.exercise_engine.start_time = time.time()
                        self.exercise_engine.exercise_start_time = time.time()
                        print("🔄 Exercise reset!")
                else:
                    # No camera, show console output
                    self._show_console_output(analysis)
                
                # Track performance
                latency = (time.perf_counter() - start_time) * 1000
                self.latency_history.append(latency)
                if len(self.latency_history) > 100:
                    self.latency_history.pop(0)
                
                # Update FPS counter
                self._update_fps_counter()
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        finally:
            self.cleanup()
    
    def _run_console_mode(self):
        """Run in console mode when OpenCV is not available"""
        print("\nRunning in console mode...")
        self.running = True
        
        try:
            while self.running:
                start_time = time.perf_counter()
                self.frame_count += 1
                
                # Get pose detection
                keypoints, frame = self.pose_detector.detect_pose_from_camera()
                
                # Analyze exercise
                analysis = self.exercise_engine.analyze_pose(keypoints)
                
                # Show console output
                self._show_console_output(analysis)
                
                # Track performance
                latency = (time.perf_counter() - start_time) * 1000
                self.latency_history.append(latency)
                if len(self.latency_history) > 100:
                    self.latency_history.pop(0)
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Simulate real-time delay
                time.sleep(0.1)  # 10 FPS for console mode
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        finally:
            self.cleanup()
    
    def _draw_ui_overlay(self, frame, analysis):
        """Draw UI overlay on frame"""
        if analysis is None:
            return frame
        
        # Draw exercise info
        exercise = analysis.get('exercise', 'Unknown').upper()
        state = analysis.get('state', 'unknown').upper()
        rep_count = analysis.get('rep_count', 0)
        quality = analysis.get('quality', 'unknown').upper()
        
        # Draw text overlay
        cv2.putText(frame, f"Exercise: {exercise}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {state}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {rep_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Quality: {quality}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw feedback
        feedback = analysis.get('feedback', '')
        if feedback:
            cv2.putText(frame, f"Feedback: {feedback}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw performance info
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Controls: S=Squat, P=Plank, R=Reset, Q=Quit", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _show_console_output(self, analysis):
        """Show analysis in console"""
        if self.frame_count % 30 == 0:  # Update every 30 frames
            print(f"\n🎯 Frame {self.frame_count} Analysis:")
            print(f"   Exercise: {analysis.get('exercise', 'Unknown').upper()}")
            print(f"   State: {analysis.get('state', 'Unknown').upper()}")
            print(f"   Reps: {analysis.get('rep_count', 0)}")
            print(f"   Quality: {analysis.get('quality', 'Unknown').upper()}")
            
            feedback = analysis.get('feedback', '')
            if feedback:
                print(f"   Feedback: {feedback}")
                print(f"   🔊 VOICE: {feedback}")
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            avg_latency = sum(self.latency_history[-10:]) / min(10, len(self.latency_history))
            print(f"⚡ Performance: {self.fps_counter} FPS, {avg_latency:.1f}ms latency")
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.pose_detector.cleanup()

def main():
    """Main entry point"""
    print("🚀 Starting EdgeCoach with Visual Display...")
    print("This version shows real-time camera feed with pose detection!")
    
    app = VisualEdgeCoach()
    app.run()

if __name__ == "__main__":
    main()
