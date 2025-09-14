"""
Exercise analysis engine for squats and planks
Provides real-time form analysis and feedback
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class ExerciseType(Enum):
    SQUAT = "squat"
    PLANK = "plank"

class SquatState(Enum):
    SETUP = "setup"
    DESCENT = "descent"
    BOTTOM = "bottom"
    ASCENT = "ascent"
    LOCKOUT = "lockout"

class PlankState(Enum):
    SETUP = "setup"
    HOLDING = "holding"
    COMPLETED = "completed"

class ExerciseEngine:
    """Main exercise analysis engine"""
    
    def __init__(self):
        self.current_exercise = None
        self.squat_state = SquatState.SETUP
        self.plank_state = PlankState.SETUP
        
        # Exercise tracking
        self.rep_count = 0
        self.start_time = None
        self.hold_start_time = None
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0  # seconds
        
        # Form analysis thresholds
        self.squat_thresholds = {
            'min_depth_angle': 90,  # degrees
            'knee_over_toe_tolerance': 0.3,  # normalized
            'back_neutral_tolerance': 15,  # degrees
            'stance_width_min': 0.8,  # normalized
            'stance_width_max': 1.2  # normalized
        }
        
        self.plank_thresholds = {
            'alignment_tolerance': 20,  # degrees
            'hip_sag_threshold': 0.1,  # normalized
            'hip_pike_threshold': 0.1,  # normalized
            'min_hold_time': 5.0  # seconds
        }
    
    def set_exercise(self, exercise_type: str):
        """Set the current exercise type"""
        try:
            self.current_exercise = ExerciseType(exercise_type.lower())
            self.reset()
            logger.info(f"Switched to {exercise_type} exercise")
        except ValueError:
            logger.error(f"Invalid exercise type: {exercise_type}")
    
    def get_current_exercise(self) -> Optional[str]:
        """Get current exercise type"""
        return self.current_exercise.value if self.current_exercise else None
    
    def reset(self):
        """Reset exercise tracking"""
        self.rep_count = 0
        self.start_time = time.time()
        self.hold_start_time = None
        self.squat_state = SquatState.SETUP
        self.plank_state = PlankState.SETUP
        logger.info("Exercise reset")
    
    def analyze_pose(self, keypoints: np.ndarray) -> Dict:
        """Analyze pose and return feedback"""
        if keypoints is None or self.current_exercise is None:
            return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
        
        if self.current_exercise == ExerciseType.SQUAT:
            return self._analyze_squat(keypoints)
        elif self.current_exercise == ExerciseType.PLANK:
            return self._analyze_plank(keypoints)
        
        return {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
    
    def _analyze_squat(self, keypoints: np.ndarray) -> Dict:
        """Analyze squat form"""
        analysis = {
            'exercise': 'squat',
            'state': self.squat_state.value,
            'rep_count': self.rep_count,
            'feedback': None,
            'quality': 'good',
            'metrics': {}
        }
        
        # Get key joint positions
        left_hip = self._get_keypoint(keypoints, 'left_hip')
        right_hip = self._get_keypoint(keypoints, 'right_hip')
        left_knee = self._get_keypoint(keypoints, 'left_knee')
        right_knee = self._get_keypoint(keypoints, 'right_knee')
        left_ankle = self._get_keypoint(keypoints, 'left_ankle')
        right_ankle = self._get_keypoint(keypoints, 'right_ankle')
        left_shoulder = self._get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self._get_keypoint(keypoints, 'right_shoulder')
        
        if not all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            analysis['feedback'] = "Position yourself in front of the camera"
            analysis['quality'] = 'unknown'
            return analysis
        
        # Calculate key metrics
        hip_center = self._get_center_point(left_hip, right_hip)
        knee_center = self._get_center_point(left_knee, right_knee)
        ankle_center = self._get_center_point(left_ankle, right_ankle)
        shoulder_center = self._get_center_point(left_shoulder, right_shoulder)
        
        # Stance width analysis
        stance_width = abs(left_ankle[0] - right_ankle[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        normalized_stance = stance_width / hip_width if hip_width > 0 else 0
        
        # Depth analysis (hip to knee angle)
        depth_angle = self._calculate_angle(hip_center, knee_center, ankle_center)
        
        # Knee over toe analysis
        knee_over_toe = abs(knee_center[0] - ankle_center[0]) / stance_width if stance_width > 0 else 0
        
        # Back alignment analysis
        back_angle = self._calculate_angle(hip_center, shoulder_center, (shoulder_center[0], shoulder_center[1] - 100))
        
        analysis['metrics'] = {
            'stance_width': normalized_stance,
            'depth_angle': depth_angle,
            'knee_over_toe': knee_over_toe,
            'back_angle': back_angle
        }
        
        # State machine for squat
        self._update_squat_state(depth_angle, normalized_stance)
        
        # Generate feedback
        feedback = self._generate_squat_feedback(
            depth_angle, normalized_stance, knee_over_toe, back_angle
        )
        
        analysis['feedback'] = feedback
        analysis['quality'] = self._assess_squat_quality(
            depth_angle, normalized_stance, knee_over_toe, back_angle
        )
        
        return analysis
    
    def _analyze_plank(self, keypoints: np.ndarray) -> Dict:
        """Analyze plank form"""
        analysis = {
            'exercise': 'plank',
            'state': self.plank_state.value,
            'rep_count': self.rep_count,
            'feedback': None,
            'quality': 'good',
            'metrics': {}
        }
        
        # Get key joint positions
        nose = self._get_keypoint(keypoints, 'nose')
        left_shoulder = self._get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self._get_keypoint(keypoints, 'right_shoulder')
        left_hip = self._get_keypoint(keypoints, 'left_hip')
        right_hip = self._get_keypoint(keypoints, 'right_hip')
        left_ankle = self._get_keypoint(keypoints, 'left_ankle')
        right_ankle = self._get_keypoint(keypoints, 'right_ankle')
        
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle]):
            analysis['feedback'] = "Position yourself in plank position"
            analysis['quality'] = 'unknown'
            return analysis
        
        # Calculate alignment metrics
        shoulder_center = self._get_center_point(left_shoulder, right_shoulder)
        hip_center = self._get_center_point(left_hip, right_hip)
        ankle_center = self._get_center_point(left_ankle, right_ankle)
        
        # Body alignment angle
        alignment_angle = self._calculate_angle(nose, shoulder_center, ankle_center)
        
        # Hip position analysis
        hip_sag = self._calculate_hip_sag(shoulder_center, hip_center, ankle_center)
        hip_pike = self._calculate_hip_pike(shoulder_center, hip_center, ankle_center)
        
        # Hold time tracking
        if self.plank_state == PlankState.SETUP and alignment_angle < 30:
            self.plank_state = PlankState.HOLDING
            self.hold_start_time = time.time()
        
        hold_time = 0
        if self.hold_start_time:
            hold_time = time.time() - self.hold_start_time
        
        analysis['metrics'] = {
            'alignment_angle': alignment_angle,
            'hip_sag': hip_sag,
            'hip_pike': hip_pike,
            'hold_time': hold_time
        }
        
        # Generate feedback
        feedback = self._generate_plank_feedback(alignment_angle, hip_sag, hip_pike, hold_time)
        
        analysis['feedback'] = feedback
        analysis['quality'] = self._assess_plank_quality(alignment_angle, hip_sag, hip_pike)
        
        return analysis
    
    def _update_squat_state(self, depth_angle: float, stance_width: float):
        """Update squat state machine"""
        if depth_angle is None or stance_width is None:
            return
        
        # Check if in proper stance
        if (stance_width < self.squat_thresholds['stance_width_min'] or 
            stance_width > self.squat_thresholds['stance_width_max']):
            self.squat_state = SquatState.SETUP
            return
        
        # State transitions based on depth
        if self.squat_state == SquatState.SETUP and depth_angle < 120:
            self.squat_state = SquatState.DESCENT
        elif self.squat_state == SquatState.DESCENT and depth_angle < self.squat_thresholds['min_depth_angle']:
            self.squat_state = SquatState.BOTTOM
        elif self.squat_state == SquatState.BOTTOM and depth_angle > self.squat_thresholds['min_depth_angle']:
            self.squat_state = SquatState.ASCENT
        elif self.squat_state == SquatState.ASCENT and depth_angle > 120:
            self.squat_state = SquatState.LOCKOUT
            self.rep_count += 1
        elif self.squat_state == SquatState.LOCKOUT and depth_angle < 120:
            self.squat_state = SquatState.DESCENT
    
    def _generate_squat_feedback(self, depth_angle: float, stance_width: float, 
                                knee_over_toe: float, back_angle: float) -> Optional[str]:
        """Generate squat feedback based on form analysis"""
        current_time = time.time()
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return None
        
        feedback = None
        
        # Stance width feedback
        if stance_width < self.squat_thresholds['stance_width_min']:
            feedback = "Widen your stance"
        elif stance_width > self.squat_thresholds['stance_width_max']:
            feedback = "Bring your feet closer together"
        
        # Depth feedback
        elif depth_angle and depth_angle > 120:
            feedback = "Go deeper - hips below knees"
        elif depth_angle and depth_angle < 60:
            feedback = "Don't go too deep"
        
        # Knee position feedback
        elif knee_over_toe > self.squat_thresholds['knee_over_toe_tolerance']:
            feedback = "Keep knees over toes"
        
        # Back alignment feedback
        elif back_angle and abs(back_angle - 90) > self.squat_thresholds['back_neutral_tolerance']:
            if back_angle < 75:
                feedback = "Chest up - don't lean forward"
            else:
                feedback = "Keep back straight"
        
        # Positive feedback
        elif (self.squat_state == SquatState.LOCKOUT and 
              stance_width >= self.squat_thresholds['stance_width_min'] and
              stance_width <= self.squat_thresholds['stance_width_max']):
            feedback = f"Great rep! Total: {self.rep_count}"
        
        if feedback:
            self.last_feedback_time = current_time
        
        return feedback
    
    def _generate_plank_feedback(self, alignment_angle: float, hip_sag: float, 
                                hip_pike: float, hold_time: float) -> Optional[str]:
        """Generate plank feedback based on form analysis"""
        current_time = time.time()
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return None
        
        feedback = None
        
        # Alignment feedback
        if alignment_angle > self.plank_thresholds['alignment_tolerance']:
            feedback = "Straighten your body - head to heels in line"
        
        # Hip position feedback
        elif hip_sag > self.plank_thresholds['hip_sag_threshold']:
            feedback = "Tuck your pelvis - don't let hips sag"
        elif hip_pike > self.plank_thresholds['hip_pike_threshold']:
            feedback = "Lower your hips - don't pike up"
        
        # Hold time feedback
        elif hold_time > 0 and hold_time < self.plank_thresholds['min_hold_time']:
            remaining = self.plank_thresholds['min_hold_time'] - hold_time
            feedback = f"Hold for {remaining:.1f} more seconds"
        elif hold_time >= self.plank_thresholds['min_hold_time']:
            feedback = f"Excellent! Hold time: {hold_time:.1f}s"
        
        if feedback:
            self.last_feedback_time = current_time
        
        return feedback
    
    def _assess_squat_quality(self, depth_angle: float, stance_width: float, 
                             knee_over_toe: float, back_angle: float) -> str:
        """Assess overall squat quality"""
        if not all([depth_angle, stance_width, knee_over_toe, back_angle]):
            return 'unknown'
        
        issues = 0
        
        if (stance_width < self.squat_thresholds['stance_width_min'] or 
            stance_width > self.squat_thresholds['stance_width_max']):
            issues += 1
        
        if depth_angle > 120:
            issues += 1
        
        if knee_over_toe > self.squat_thresholds['knee_over_toe_tolerance']:
            issues += 1
        
        if abs(back_angle - 90) > self.squat_thresholds['back_neutral_tolerance']:
            issues += 1
        
        if issues == 0:
            return 'excellent'
        elif issues <= 1:
            return 'good'
        elif issues <= 2:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_plank_quality(self, alignment_angle: float, hip_sag: float, hip_pike: float) -> str:
        """Assess overall plank quality"""
        if not all([alignment_angle, hip_sag, hip_pike]):
            return 'unknown'
        
        issues = 0
        
        if alignment_angle > self.plank_thresholds['alignment_tolerance']:
            issues += 1
        
        if hip_sag > self.plank_thresholds['hip_sag_threshold']:
            issues += 1
        
        if hip_pike > self.plank_thresholds['hip_pike_threshold']:
            issues += 1
        
        if issues == 0:
            return 'excellent'
        elif issues <= 1:
            return 'good'
        else:
            return 'fair'
    
    def _get_keypoint(self, keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float, float]]:
        """Get keypoint by name (simplified version)"""
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        try:
            index = keypoint_names.index(name)
            if index < len(keypoints):
                return tuple(keypoints[index])
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _get_center_point(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float]:
        """Get center point between two keypoints"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def _calculate_angle(self, p1: Tuple[float, float], vertex: Tuple[float, float], p3: Tuple[float, float]) -> Optional[float]:
        """Calculate angle between three points"""
        try:
            a = np.array([p1[0], p1[1]])
            b = np.array([vertex[0], vertex[1]])
            c = np.array([p3[0], p3[1]])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except:
            return None
    
    def _calculate_hip_sag(self, shoulder: Tuple[float, float], hip: Tuple[float, float], ankle: Tuple[float, float]) -> float:
        """Calculate hip sag in plank position"""
        # Simplified sag calculation - distance below shoulder-ankle line
        line_y = shoulder[1] + (ankle[1] - shoulder[1]) * (hip[0] - shoulder[0]) / (ankle[0] - shoulder[0])
        sag = max(0, hip[1] - line_y)
        return sag / (ankle[1] - shoulder[1]) if ankle[1] != shoulder[1] else 0
    
    def _calculate_hip_pike(self, shoulder: Tuple[float, float], hip: Tuple[float, float], ankle: Tuple[float, float]) -> float:
        """Calculate hip pike in plank position"""
        # Simplified pike calculation - distance above shoulder-ankle line
        line_y = shoulder[1] + (ankle[1] - shoulder[1]) * (hip[0] - shoulder[0]) / (ankle[0] - shoulder[0])
        pike = max(0, line_y - hip[1])
        return pike / (ankle[1] - shoulder[1]) if ankle[1] != shoulder[1] else 0
