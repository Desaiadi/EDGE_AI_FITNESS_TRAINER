#!/usr/bin/env python3
"""
EdgeCoach Interactive Demo - Shows how the system works
Interactive version with keyboard controls and realistic exercise simulation
"""

import time
import random
import logging
import threading
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractivePoseEstimator:
    """Interactive pose estimator that simulates realistic pose detection"""
    
    def __init__(self):
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Simulate different exercise poses
        self.squat_poses = self._generate_squat_poses()
        self.plank_poses = self._generate_plank_poses()
        self.current_pose_index = 0
    
    def _generate_squat_poses(self):
        """Generate realistic squat pose sequences"""
        poses = []
        
        # Setup pose - standing
        poses.append({
            'name': 'Setup',
            'keypoints': self._create_standing_pose(),
            'description': 'Standing ready position'
        })
        
        # Descent poses
        for i in range(5):
            poses.append({
                'name': 'Descent',
                'keypoints': self._create_squat_pose(90 + i * 15),  # 90 to 150 degrees
                'description': f'Going down - {90 + i * 15}°'
            })
        
        # Bottom pose
        poses.append({
            'name': 'Bottom',
            'keypoints': self._create_squat_pose(85),
            'description': 'Deep squat position'
        })
        
        # Ascent poses
        for i in range(5):
            poses.append({
                'name': 'Ascent',
                'keypoints': self._create_squat_pose(85 + i * 15),  # 85 to 145 degrees
                'description': f'Coming up - {85 + i * 15}°'
            })
        
        # Lockout pose
        poses.append({
            'name': 'Lockout',
            'keypoints': self._create_standing_pose(),
            'description': 'Standing tall - rep complete!'
        })
        
        return poses
    
    def _generate_plank_poses(self):
        """Generate realistic plank pose sequences"""
        poses = []
        
        # Setup pose
        poses.append({
            'name': 'Setup',
            'keypoints': self._create_plank_pose(0, 0),
            'description': 'Getting into plank position'
        })
        
        # Good plank poses
        for i in range(8):
            poses.append({
                'name': 'Holding',
                'keypoints': self._create_plank_pose(5 + random.uniform(-3, 3), 0.05 + random.uniform(-0.02, 0.02)),
                'description': f'Good plank hold - {i+1}s'
            })
        
        # Some form issues
        poses.append({
            'name': 'Hip Sag',
            'keypoints': self._create_plank_pose(15, 0.12),
            'description': 'Hips sagging - needs correction'
        })
        
        poses.append({
            'name': 'Hip Pike',
            'keypoints': self._create_plank_pose(25, -0.08),
            'description': 'Hips too high - pike position'
        })
        
        # Back to good form
        for i in range(3):
            poses.append({
                'name': 'Holding',
                'keypoints': self._create_plank_pose(5 + random.uniform(-2, 2), 0.05 + random.uniform(-0.01, 0.01)),
                'description': f'Corrected form - {i+1}s'
            })
        
        return poses
    
    def _create_standing_pose(self):
        """Create a standing pose"""
        keypoints = []
        # Nose
        keypoints.append([320, 100, 0.95])
        # Eyes
        keypoints.append([310, 90, 0.9])
        keypoints.append([330, 90, 0.9])
        # Ears
        keypoints.append([300, 95, 0.85])
        keypoints.append([340, 95, 0.85])
        # Shoulders
        keypoints.append([280, 150, 0.9])
        keypoints.append([360, 150, 0.9])
        # Elbows
        keypoints.append([250, 200, 0.8])
        keypoints.append([390, 200, 0.8])
        # Wrists
        keypoints.append([220, 250, 0.75])
        keypoints.append([420, 250, 0.75])
        # Hips
        keypoints.append([300, 250, 0.9])
        keypoints.append([340, 250, 0.9])
        # Knees
        keypoints.append([300, 350, 0.85])
        keypoints.append([340, 350, 0.85])
        # Ankles
        keypoints.append([300, 450, 0.8])
        keypoints.append([340, 450, 0.8])
        
        return keypoints
    
    def _create_squat_pose(self, hip_knee_angle):
        """Create a squat pose with specific hip-knee angle"""
        keypoints = []
        # Nose
        keypoints.append([320, 100, 0.95])
        # Eyes
        keypoints.append([310, 90, 0.9])
        keypoints.append([330, 90, 0.9])
        # Ears
        keypoints.append([300, 95, 0.85])
        keypoints.append([340, 95, 0.85])
        # Shoulders
        keypoints.append([280, 150, 0.9])
        keypoints.append([360, 150, 0.9])
        # Elbows
        keypoints.append([250, 200, 0.8])
        keypoints.append([390, 200, 0.8])
        # Wrists
        keypoints.append([220, 250, 0.75])
        keypoints.append([420, 250, 0.75])
        
        # Hips - move down based on angle
        hip_y = 250 + (150 - hip_knee_angle) * 2
        keypoints.append([300, hip_y, 0.9])
        keypoints.append([340, hip_y, 0.9])
        
        # Knees - move down and forward
        knee_y = hip_y + 80
        knee_forward = (150 - hip_knee_angle) * 0.5
        keypoints.append([300 + knee_forward, knee_y, 0.85])
        keypoints.append([340 + knee_forward, knee_y, 0.85])
        
        # Ankles
        keypoints.append([300, 450, 0.8])
        keypoints.append([340, 450, 0.8])
        
        return keypoints
    
    def _create_plank_pose(self, alignment_angle, hip_sag):
        """Create a plank pose with specific alignment and hip position"""
        keypoints = []
        # Nose
        keypoints.append([320, 200, 0.95])
        # Eyes
        keypoints.append([310, 190, 0.9])
        keypoints.append([330, 190, 0.9])
        # Ears
        keypoints.append([300, 195, 0.85])
        keypoints.append([340, 195, 0.85])
        # Shoulders
        keypoints.append([280, 220, 0.9])
        keypoints.append([360, 220, 0.9])
        # Elbows
        keypoints.append([250, 250, 0.8])
        keypoints.append([390, 250, 0.8])
        # Wrists
        keypoints.append([220, 280, 0.75])
        keypoints.append([420, 280, 0.75])
        
        # Hips - adjust based on sag
        hip_y = 280 + hip_sag * 100
        keypoints.append([300, hip_y, 0.9])
        keypoints.append([340, hip_y, 0.9])
        
        # Knees
        keypoints.append([300, 350, 0.85])
        keypoints.append([340, 350, 0.85])
        
        # Ankles
        keypoints.append([300, 420, 0.8])
        keypoints.append([340, 420, 0.8])
        
        return keypoints
    
    def estimate_pose(self, exercise_type=None):
        """Estimate pose based on current exercise"""
        if exercise_type == 'squat':
            pose = self.squat_poses[self.current_pose_index % len(self.squat_poses)]
            self.current_pose_index += 1
            return pose['keypoints'], pose['name'], pose['description']
        elif exercise_type == 'plank':
            pose = self.plank_poses[self.current_pose_index % len(self.plank_poses)]
            self.current_pose_index += 1
            return pose['keypoints'], pose['name'], pose['description']
        else:
            return self._create_standing_pose(), 'Unknown', 'No exercise selected'

class InteractiveExerciseEngine:
    """Interactive exercise engine with realistic analysis"""
    
    def __init__(self):
        self.current_exercise = None
        self.rep_count = 0
        self.start_time = time.time()
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0
        self.exercise_start_time = None
        
    def set_exercise(self, exercise_type: str):
        """Set the current exercise type"""
        self.current_exercise = exercise_type
        self.rep_count = 0
        self.exercise_start_time = time.time()
        print(f"\n🏃‍♂️ Switched to {exercise_type.upper()} exercise!")
        print("   Follow the on-screen instructions and try the movements!")
    
    def analyze_pose(self, keypoints, pose_name, pose_description):
        """Analyze pose and return feedback"""
        if keypoints is None or self.current_exercise is None:
            return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
        
        current_time = time.time()
        
        if self.current_exercise == 'squat':
            return self._analyze_squat_interactive(keypoints, pose_name, pose_description, current_time)
        elif self.current_exercise == 'plank':
            return self._analyze_plank_interactive(keypoints, pose_name, pose_description, current_time)
        
        return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
    
    def _analyze_squat_interactive(self, keypoints, pose_name, pose_description, current_time):
        """Interactive squat analysis"""
        # Extract key metrics from pose
        left_hip = keypoints[11]  # left_hip
        right_hip = keypoints[12]  # right_hip
        left_knee = keypoints[13]  # left_knee
        right_knee = keypoints[14]  # right_knee
        left_ankle = keypoints[15]  # left_ankle
        right_ankle = keypoints[16]  # right_ankle
        
        # Calculate metrics
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        knee_center_y = (left_knee[1] + right_knee[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        depth_angle = 180 - abs(hip_center_y - knee_center_y) * 0.5
        stance_width = abs(left_ankle[0] - right_ankle[0]) / 100
        
        feedback = None
        quality = 'good'
        
        # Generate feedback based on pose name and metrics
        if pose_name == 'Setup':
            feedback = "Get ready! Stand with feet shoulder-width apart"
            quality = 'good'
        elif pose_name == 'Descent':
            if depth_angle > 120:
                feedback = "Go deeper! Hips should go below knees"
                quality = 'fair'
            else:
                feedback = "Good depth! Keep going down slowly"
                quality = 'good'
        elif pose_name == 'Bottom':
            if depth_angle < 90:
                feedback = "Perfect depth! Now drive up through your heels"
                quality = 'excellent'
            else:
                feedback = "Almost there! Go a bit deeper"
                quality = 'fair'
        elif pose_name == 'Ascent':
            feedback = "Great! Drive up through your heels, chest up"
            quality = 'good'
        elif pose_name == 'Lockout':
            self.rep_count += 1
            feedback = f"Excellent rep! Total: {self.rep_count} squats"
            quality = 'excellent'
        
        return {
            'exercise': 'squat',
            'state': pose_name.lower(),
            'rep_count': self.rep_count,
            'feedback': feedback,
            'quality': quality,
            'metrics': {
                'depth_angle': depth_angle,
                'stance_width': stance_width,
                'pose_description': pose_description
            }
        }
    
    def _analyze_plank_interactive(self, keypoints, pose_name, pose_description, current_time):
        """Interactive plank analysis"""
        # Extract key metrics
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Calculate metrics
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        alignment_angle = abs(hip_center_y - shoulder_center_y) * 0.3
        hip_sag = max(0, hip_center_y - shoulder_center_y) / 100
        hold_time = current_time - self.exercise_start_time if self.exercise_start_time else 0
        
        feedback = None
        quality = 'good'
        
        # Generate feedback based on pose
        if pose_name == 'Setup':
            feedback = "Get into plank position! Hands under shoulders"
            quality = 'good'
        elif pose_name == 'Holding':
            if alignment_angle < 10:
                feedback = f"Perfect plank! Hold for {int(5 - hold_time)} more seconds"
                quality = 'excellent'
            else:
                feedback = "Good hold! Keep your body straight"
                quality = 'good'
        elif pose_name == 'Hip Sag':
            feedback = "Hips are sagging! Tuck your pelvis and engage your core"
            quality = 'fair'
        elif pose_name == 'Hip Pike':
            feedback = "Hips too high! Lower them to create a straight line"
            quality = 'fair'
        
        return {
            'exercise': 'plank',
            'state': pose_name.lower(),
            'rep_count': int(hold_time // 10),  # Count 10-second intervals
            'feedback': feedback,
            'quality': quality,
            'metrics': {
                'alignment_angle': alignment_angle,
                'hip_sag': hip_sag,
                'hold_time': hold_time,
                'pose_description': pose_description
            }
        }

class InteractiveVoiceCoach:
    """Interactive voice coach with realistic feedback"""
    
    def __init__(self):
        self.speaking = False
        self.last_speech_time = 0
        self.min_speech_interval = 1.5
    
    def speak(self, message: str):
        """Simulate voice feedback with realistic timing"""
        if not message:
            return
        
        current_time = time.time()
        if current_time - self.last_speech_time < self.min_speech_interval:
            return
        
        # Simulate voice output with realistic delay
        print(f"🔊 VOICE: {message}")
        self.last_speech_time = current_time

class InteractiveUIOverlay:
    """Interactive UI overlay with detailed information"""
    
    def __init__(self):
        self.current_analysis = None
        self.frame_count = 0
    
    def update_analysis(self, analysis: dict):
        """Update current analysis data"""
        self.current_analysis = analysis
        self.frame_count += 1
    
    def draw_overlay(self):
        """Draw detailed UI overlay"""
        if self.current_analysis is None:
            return
        
        # Clear screen (simplified)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print("🎯 EDGECOACH - AI FITNESS COACH (Interactive Demo)")
        print("=" * 80)
        
        # Exercise info
        exercise = self.current_analysis.get('exercise', 'Unknown').upper()
        state = self.current_analysis.get('state', 'unknown').upper()
        rep_count = self.current_analysis.get('rep_count', 0)
        quality = self.current_analysis.get('quality', 'unknown').upper()
        
        print(f"Exercise: {exercise}")
        print(f"State: {state}")
        print(f"Reps: {rep_count}")
        print(f"Quality: {quality}")
        
        # Feedback
        feedback = self.current_analysis.get('feedback', '')
        if feedback:
            print(f"\n💬 Feedback: {feedback}")
        
        # Metrics
        metrics = self.current_analysis.get('metrics', {})
        if metrics:
            print(f"\n📊 Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.1f}")
                else:
                    print(f"   {key}: {value}")
        
        # Performance info
        print(f"\n⚡ Performance: Frame {self.frame_count}")
        print("=" * 80)
        print("Controls: S=Squat, P=Plank, R=Reset, Q=Quit")
        print("=" * 80)

class InteractiveEdgeCoach:
    """Interactive EdgeCoach application"""
    
    def __init__(self):
        self.running = False
        self.pose_estimator = InteractivePoseEstimator()
        self.exercise_engine = InteractiveExerciseEngine()
        self.voice_coach = InteractiveVoiceCoach()
        self.ui_overlay = InteractiveUIOverlay()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_history = []
        
        # Input handling
        self.input_thread = None
        self.user_input = None
    
    def start_input_thread(self):
        """Start input handling thread"""
        self.input_thread = threading.Thread(target=self._handle_input, daemon=True)
        self.input_thread.start()
    
    def _handle_input(self):
        """Handle user input in background thread"""
        while self.running:
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    self._process_keypress(key)
            except:
                # Fallback for Windows
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    self._process_keypress(key)
                time.sleep(0.1)
    
    def _process_keypress(self, key: str):
        """Process key presses"""
        if key == 'q':
            self.running = False
            print("\n👋 Thanks for trying EdgeCoach!")
        elif key == 's':
            self.exercise_engine.set_exercise('squat')
            self.pose_estimator.current_pose_index = 0
        elif key == 'p':
            self.exercise_engine.set_exercise('plank')
            self.pose_estimator.current_pose_index = 0
        elif key == 'r':
            self.exercise_engine.rep_count = 0
            self.exercise_engine.start_time = time.time()
            self.exercise_engine.exercise_start_time = time.time()
            self.pose_estimator.current_pose_index = 0
            print("🔄 Exercise reset!")
    
    def run(self):
        """Main application loop"""
        print("🚀 Starting EdgeCoach Interactive Demo...")
        print("This demo shows how EdgeCoach analyzes your exercise form in real-time!")
        print("\nPress 'S' for Squat, 'P' for Plank, 'R' to Reset, 'Q' to Quit")
        print("Watch how the system provides feedback as you 'perform' the exercises!")
        
        self.running = True
        self.start_input_thread()
        
        try:
            while self.running:
                # Simulate frame processing
                start_time = time.perf_counter()
                
                # Get pose estimation
                keypoints, pose_name, pose_description = self.pose_estimator.estimate_pose(
                    self.exercise_engine.current_exercise
                )
                
                # Analyze exercise
                analysis = self.exercise_engine.analyze_pose(keypoints, pose_name, pose_description)
                
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
                time.sleep(0.5)  # 2 FPS for interactive demo
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
        finally:
            self.cleanup()
    
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
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)

def main():
    """Main entry point"""
    print("🎯 EdgeCoach Interactive Demo")
    print("=" * 50)
    print("This demo shows how EdgeCoach works:")
    print("1. Real-time pose estimation")
    print("2. Exercise form analysis")
    print("3. Voice feedback system")
    print("4. Visual UI overlay")
    print("5. Performance monitoring")
    print("=" * 50)
    
    app = InteractiveEdgeCoach()
    app.run()

if __name__ == "__main__":
    main()
