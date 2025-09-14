#!/usr/bin/env python3
"""
EdgeCoach with ASCII Visual Display
Shows real-time pose detection using ASCII art visualization
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

class ASCIIPoseDetector:
    """ASCII pose detector with visual display"""
    
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
        
        # ASCII display parameters
        self.display_width = 80
        self.display_height = 24
        
    def detect_pose(self):
        """Detect pose and return keypoints"""
        # Simulate pose detection with realistic movement
        keypoints = self._generate_realistic_pose()
        return keypoints
    
    def _generate_realistic_pose(self):
        """Generate realistic pose with movement"""
        # Base pose centered in display
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Add some realistic movement
        movement = random.uniform(-5, 5)
        
        keypoints = []
        
        # Head region
        keypoints.append([center_x + movement, center_y - 8, 0.9])  # nose
        keypoints.append([center_x - 2 + movement, center_y - 9, 0.8])  # left_eye
        keypoints.append([center_x + 2 + movement, center_y - 9, 0.8])  # right_eye
        keypoints.append([center_x - 4 + movement, center_y - 8, 0.7])  # left_ear
        keypoints.append([center_x + 4 + movement, center_y - 8, 0.7])  # right_ear
        
        # Shoulder region
        keypoints.append([center_x - 8 + movement, center_y - 4, 0.9])  # left_shoulder
        keypoints.append([center_x + 8 + movement, center_y - 4, 0.9])  # right_shoulder
        keypoints.append([center_x - 12 + movement, center_y - 2, 0.8])  # left_elbow
        keypoints.append([center_x + 12 + movement, center_y - 2, 0.8])  # right_elbow
        keypoints.append([center_x - 16 + movement, center_y, 0.7])  # left_wrist
        keypoints.append([center_x + 16 + movement, center_y, 0.7])  # right_wrist
        
        # Hip region
        keypoints.append([center_x - 6 + movement, center_y + 2, 0.9])  # left_hip
        keypoints.append([center_x + 6 + movement, center_y + 2, 0.9])  # right_hip
        keypoints.append([center_x - 7 + movement, center_y + 6, 0.8])  # left_knee
        keypoints.append([center_x + 7 + movement, center_y + 6, 0.8])  # right_knee
        keypoints.append([center_x - 6 + movement, center_y + 10, 0.7])  # left_ankle
        keypoints.append([center_x + 6 + movement, center_y + 10, 0.7])  # right_ankle
        
        return keypoints
    
    def draw_ascii_skeleton(self, keypoints, analysis=None):
        """Draw ASCII skeleton representation"""
        # Create display grid
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # Only draw confident keypoints
                # Convert to grid coordinates
                grid_x = int(x) % self.display_width
                grid_y = int(y) % self.display_height
                
                if 0 <= grid_x < self.display_width and 0 <= grid_y < self.display_height:
                    if conf > 0.8:
                        grid[grid_y][grid_x] = '●'  # High confidence
                    else:
                        grid[grid_y][grid_x] = '○'  # Medium confidence
        
        # Draw skeleton connections
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        
        for start_idx, end_idx in skeleton_connections:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            if (start_point[2] > 0.5 and end_point[2] > 0.5):
                # Draw line between points
                self._draw_line(grid, start_point, end_point)
        
        # Display the grid
        print("\n" + "=" * self.display_width)
        print("🎯 REAL-TIME POSE DETECTION")
        print("=" * self.display_width)
        
        for row in grid:
            print(''.join(row))
        
        print("=" * self.display_width)
        
        # Show keypoint information
        if analysis:
            print(f"Exercise: {analysis.get('exercise', 'Unknown').upper()}")
            print(f"State: {analysis.get('state', 'Unknown').upper()}")
            print(f"Reps: {analysis.get('rep_count', 0)}")
            print(f"Quality: {analysis.get('quality', 'Unknown').upper()}")
            
            feedback = analysis.get('feedback', '')
            if feedback:
                print(f"Feedback: {feedback}")
                print(f"🔊 VOICE: {feedback}")
        
        print("=" * self.display_width)
    
    def _draw_line(self, grid, start_point, end_point):
        """Draw line between two points in ASCII grid"""
        x1, y1 = int(start_point[0]), int(start_point[1])
        x2, y2 = int(end_point[0]), int(end_point[1])
        
        # Simple line drawing algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > dy:
            # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                y = int(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    if grid[y][x] == ' ':
                        grid[y][x] = '-'
        else:
            # Vertical line
            for y in range(min(y1, y2), max(y1, y2) + 1):
                x = int(x1 + (x2 - x1) * (y - y1) / (y2 - y1))
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    if grid[y][x] == ' ':
                        grid[y][x] = '|'

class ASCIIExerciseEngine:
    """ASCII exercise engine with real-time analysis"""
    
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
        print("   Watch the ASCII skeleton and follow the feedback!")
    
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

class ASCIIEdgeCoach:
    """Main EdgeCoach application with ASCII display"""
    
    def __init__(self):
        self.running = False
        self.pose_detector = ASCIIPoseDetector()
        self.exercise_engine = ASCIIExerciseEngine()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_history = []
        self.frame_count = 0
    
    def run(self):
        """Main application loop with ASCII display"""
        print("🎯 EdgeCoach with ASCII Visual Display")
        print("=" * 50)
        print("This version shows real-time pose detection using ASCII art!")
        print("Watch the skeleton move and follow the feedback!")
        
        print("\nControls:")
        print("  S - Switch to Squat exercise")
        print("  P - Switch to Plank exercise")
        print("  R - Reset exercise")
        print("  Q - Quit application")
        print("\nPress any key to start...")
        
        self.running = True
        
        try:
            while self.running:
                # Process frame
                start_time = time.perf_counter()
                self.frame_count += 1
                
                # Get pose detection
                keypoints = self.pose_detector.detect_pose()
                
                # Analyze exercise
                analysis = self.exercise_engine.analyze_pose(keypoints)
                
                # Draw ASCII skeleton
                self.pose_detector.draw_ascii_skeleton(keypoints, analysis)
                
                # Track performance
                latency = (time.perf_counter() - start_time) * 1000
                self.latency_history.append(latency)
                if len(self.latency_history) > 100:
                    self.latency_history.pop(0)
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Simulate real-time delay
                time.sleep(0.5)  # 2 FPS for ASCII display
                
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        finally:
            self.cleanup()
    
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
        print("\n🎉 EdgeCoach ASCII Demo Complete!")
        print("This showed how EdgeCoach works with real-time pose detection!")

def main():
    """Main entry point"""
    print("🚀 Starting EdgeCoach with ASCII Visual Display...")
    print("This version shows real-time pose detection using ASCII art!")
    
    app = ASCIIEdgeCoach()
    app.run()

if __name__ == "__main__":
    main()
