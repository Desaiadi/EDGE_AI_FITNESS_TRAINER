#!/usr/bin/env python3
"""
EdgeCoach Simple Demo - Works without heavy dependencies
Demonstrates the core functionality without ONNX Runtime and OpenCV
"""

import time
import random
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePoseEstimator:
    """Simplified pose estimator for demo purposes"""
    
    def __init__(self):
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def estimate_pose(self, frame_data=None):
        """Generate dummy pose data for demo"""
        # Simulate pose detection with random but realistic data
        keypoints = []
        for i in range(17):
            # Generate realistic keypoint positions
            x = random.uniform(200, 400) + random.uniform(-50, 50)
            y = random.uniform(100, 400) + random.uniform(-50, 50)
            confidence = random.uniform(0.7, 0.95)
            keypoints.append([x, y, confidence])
        
        return keypoints

class SimpleExerciseEngine:
    """Simplified exercise engine for demo"""
    
    def __init__(self):
        self.current_exercise = None
        self.rep_count = 0
        self.start_time = time.time()
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0
        
    def set_exercise(self, exercise_type: str):
        """Set the current exercise type"""
        self.current_exercise = exercise_type
        self.rep_count = 0
        logger.info(f"Switched to {exercise_type} exercise")
    
    def analyze_pose(self, keypoints):
        """Analyze pose and return feedback"""
        if keypoints is None or self.current_exercise is None:
            return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
        
        # Simulate exercise analysis
        current_time = time.time()
        
        if self.current_exercise == 'squat':
            return self._analyze_squat_demo(keypoints, current_time)
        elif self.current_exercise == 'plank':
            return self._analyze_plank_demo(keypoints, current_time)
        
        return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
    
    def _analyze_squat_demo(self, keypoints, current_time):
        """Demo squat analysis"""
        # Simulate random form analysis
        depth_angle = random.uniform(60, 120)
        stance_width = random.uniform(0.8, 1.2)
        knee_over_toe = random.uniform(0.1, 0.4)
        
        feedback = None
        quality = 'good'
        
        # Generate feedback based on simulated metrics
        if depth_angle > 110:
            feedback = "Go deeper - hips below knees"
            quality = 'fair'
        elif stance_width < 0.9:
            feedback = "Widen your stance"
            quality = 'fair'
        elif knee_over_toe > 0.3:
            feedback = "Keep knees over toes"
            quality = 'fair'
        elif random.random() < 0.1:  # 10% chance of good feedback
            self.rep_count += 1
            feedback = f"Great rep! Total: {self.rep_count}"
            quality = 'excellent'
        
        return {
            'exercise': 'squat',
            'state': 'active',
            'rep_count': self.rep_count,
            'feedback': feedback,
            'quality': quality,
            'metrics': {
                'depth_angle': depth_angle,
                'stance_width': stance_width,
                'knee_over_toe': knee_over_toe
            }
        }
    
    def _analyze_plank_demo(self, keypoints, current_time):
        """Demo plank analysis"""
        # Simulate plank analysis
        alignment_angle = random.uniform(5, 25)
        hip_sag = random.uniform(0.05, 0.15)
        hold_time = current_time - self.start_time
        
        feedback = None
        quality = 'good'
        
        if alignment_angle > 20:
            feedback = "Straighten your body - head to heels in line"
            quality = 'fair'
        elif hip_sag > 0.1:
            feedback = "Tuck your pelvis - don't let hips sag"
            quality = 'fair'
        elif hold_time < 5:
            remaining = 5 - hold_time
            feedback = f"Hold for {remaining:.1f} more seconds"
        else:
            feedback = f"Excellent! Hold time: {hold_time:.1f}s"
            quality = 'excellent'
        
        return {
            'exercise': 'plank',
            'state': 'holding',
            'rep_count': int(hold_time // 10),  # Count 10-second intervals
            'feedback': feedback,
            'quality': quality,
            'metrics': {
                'alignment_angle': alignment_angle,
                'hip_sag': hip_sag,
                'hold_time': hold_time
            }
        }

class SimpleVoiceCoach:
    """Simplified voice coach for demo"""
    
    def __init__(self):
        self.speaking = False
        self.last_speech_time = 0
        self.min_speech_interval = 1.0
    
    def speak(self, message: str):
        """Simulate voice feedback"""
        if not message:
            return
        
        current_time = time.time()
        if current_time - self.last_speech_time < self.min_speech_interval:
            return
        
        # Simulate voice output
        print(f"🔊 VOICE: {message}")
        self.last_speech_time = current_time

class SimpleUIOverlay:
    """Simplified UI overlay for demo"""
    
    def __init__(self):
        self.current_analysis = None
    
    def update_analysis(self, analysis: Dict):
        """Update current analysis data"""
        self.current_analysis = analysis
    
    def draw_overlay(self, frame_data=None):
        """Draw UI overlay (simulated)"""
        if self.current_analysis is None:
            return
        
        print("\n" + "="*60)
        print("🎯 EDGECOACH - AI FITNESS COACH")
        print("="*60)
        
        # Exercise info
        exercise = self.current_analysis.get('exercise', 'Unknown').upper()
        rep_count = self.current_analysis.get('rep_count', 0)
        quality = self.current_analysis.get('quality', 'unknown').upper()
        
        print(f"Exercise: {exercise}")
        print(f"Reps: {rep_count}")
        print(f"Quality: {quality}")
        
        # Feedback
        feedback = self.current_analysis.get('feedback', '')
        if feedback:
            print(f"Feedback: {feedback}")
        
        # Metrics
        metrics = self.current_analysis.get('metrics', {})
        if metrics:
            print("\nMetrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.1f}")
                else:
                    print(f"  {key}: {value}")
        
        print("="*60)
        print("Controls: S=Squat, P=Plank, R=Reset, Q=Quit")
        print("="*60)

class SimpleEdgeCoach:
    """Simplified EdgeCoach application for demo"""
    
    def __init__(self):
        self.running = False
        self.pose_estimator = SimplePoseEstimator()
        self.exercise_engine = SimpleExerciseEngine()
        self.voice_coach = SimpleVoiceCoach()
        self.ui_overlay = SimpleUIOverlay()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_history = []
    
    def run(self):
        """Main application loop"""
        logger.info("EdgeCoach Simple Demo Started")
        logger.info("This is a simplified version that works without heavy dependencies")
        
        self.running = True
        
        print("\n🎉 Welcome to EdgeCoach Simple Demo!")
        print("This demo simulates the full EdgeCoach experience")
        print("Press 'S' for Squat, 'P' for Plank, 'R' to Reset, 'Q' to Quit")
        
        try:
            while self.running:
                # Simulate frame processing
                start_time = time.perf_counter()
                
                # Get pose estimation
                keypoints = self.pose_estimator.estimate_pose()
                
                # Analyze exercise
                analysis = self.exercise_engine.analyze_pose(keypoints)
                
                # Generate voice feedback
                if analysis.get('feedback'):
                    self.voice_coach.speak(analysis['feedback'])
                
                # Update UI
                self.ui_overlay.update_analysis(analysis)
                self.ui_overlay.draw_overlay()
                
                # Track performance
                latency = (time.perf_counter() - start_time) * 1000
                self.latency_history.append(latency)
                if len(self.latency_history) > 100:
                    self.latency_history.pop(0)
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Simulate real-time processing delay
                time.sleep(0.1)  # 10 FPS for demo
                
                # Check for user input (simplified)
                try:
                    import sys
                    import select
                    import tty
                    import termios
                    
                    # Non-blocking input check
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1).lower()
                        self._handle_keypress(key)
                except:
                    # Fallback for Windows
                    pass
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            self.cleanup()
    
    def _handle_keypress(self, key: str):
        """Handle key presses"""
        if key == 'q':
            self.running = False
            print("\n👋 Thanks for trying EdgeCoach!")
        elif key == 's':
            self.exercise_engine.set_exercise('squat')
        elif key == 'p':
            self.exercise_engine.set_exercise('plank')
        elif key == 'r':
            self.exercise_engine.rep_count = 0
            self.exercise_engine.start_time = time.time()
            print("🔄 Exercise reset")
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            avg_latency = sum(self.latency_history[-10:]) / min(10, len(self.latency_history))
            print(f"\n📊 Performance: {self.fps_counter} FPS, {avg_latency:.1f}ms latency")
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        logger.info("EdgeCoach Simple Demo cleanup completed")

def main():
    """Main entry point"""
    print("🚀 Starting EdgeCoach Simple Demo...")
    print("Note: This is a simplified version for demonstration purposes")
    print("The full version requires ONNX Runtime and OpenCV for real pose detection")
    
    app = SimpleEdgeCoach()
    app.run()

if __name__ == "__main__":
    main()
