"""
Pose estimation module using ONNX Runtime with DirectML
Supports MoveNet and BlazePose models for real-time pose detection
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import Optional, Tuple, List, Dict
import time

logger = logging.getLogger(__name__)

class PoseEstimator:
    """Real-time pose estimation using ONNX Runtime with DirectML"""
    
    def __init__(self, model_path: str = "models/movenet_singlepose_lightning_4.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (192, 192)  # MoveNet Lightning input size
        
        # Pose keypoint indices (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Key indices for exercise analysis
        self.hip_indices = [11, 12]  # left_hip, right_hip
        self.knee_indices = [13, 14]  # left_knee, right_knee
        self.ankle_indices = [15, 16]  # left_ankle, right_ankle
        self.shoulder_indices = [5, 6]  # left_shoulder, right_shoulder
        self.nose_index = 0
        
    def initialize(self) -> bool:
        """Initialize the ONNX model with DirectML provider"""
        try:
            # Configure ONNX Runtime providers
            providers = [
                'DmlExecutionProvider',  # DirectML for NPU/GPU
                'CPUExecutionProvider'   # Fallback to CPU
            ]
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Log provider information
            logger.info(f"ONNX Runtime providers: {self.session.get_providers()}")
            logger.info(f"Input name: {self.input_name}, Output name: {self.output_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pose estimator: {e}")
            return False
    
    def estimate_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose from input frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Keypoints array with shape (17, 3) [x, y, confidence] or None
        """
        if self.session is None:
            return None
        
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            start_time = time.perf_counter()
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # Postprocess results
            keypoints = self._postprocess_outputs(outputs[0], frame.shape)
            
            # Log performance
            if inference_time > 50:  # Log slow inferences
                logger.warning(f"Slow inference: {inference_time:.1f}ms")
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return None
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input"""
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)
        
        return input_tensor
    
    def _postprocess_outputs(self, outputs: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """Postprocess model outputs to keypoints"""
        # MoveNet outputs keypoints in normalized coordinates [0, 1]
        # Shape: (1, 1, 17, 3) -> (17, 3)
        keypoints = outputs[0, 0]  # Remove batch and detection dimensions
        
        # Convert normalized coordinates to pixel coordinates
        height, width = frame_shape[:2]
        keypoints[:, 0] *= width   # x coordinates
        keypoints[:, 1] *= height  # y coordinates
        
        # Keep confidence scores as is
        # keypoints[:, 2] remains as confidence
        
        return keypoints
    
    def get_keypoint(self, keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float, float]]:
        """Get specific keypoint by name"""
        if keypoints is None:
            return None
        
        try:
            index = self.keypoint_names.index(name)
            return tuple(keypoints[index])
        except ValueError:
            logger.warning(f"Unknown keypoint name: {name}")
            return None
    
    def get_confidence(self, keypoints: np.ndarray, name: str) -> float:
        """Get confidence score for specific keypoint"""
        keypoint = self.get_keypoint(keypoints, name)
        return keypoint[2] if keypoint else 0.0
    
    def is_keypoint_visible(self, keypoints: np.ndarray, name: str, threshold: float = 0.3) -> bool:
        """Check if keypoint is visible based on confidence threshold"""
        return self.get_confidence(keypoints, name) > threshold
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, 
                     confidence_threshold: float = 0.3) -> np.ndarray:
        """Draw skeleton overlay on frame"""
        if keypoints is None:
            return frame
        
        # Define skeleton connections (COCO format)
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
    
    def calculate_angle(self, keypoints: np.ndarray, joint1: str, joint2: str, joint3: str) -> Optional[float]:
        """Calculate angle between three joints"""
        p1 = self.get_keypoint(keypoints, joint1)
        p2 = self.get_keypoint(keypoints, joint2)  # Vertex
        p3 = self.get_keypoint(keypoints, joint3)
        
        if not all([p1, p2, p3]):
            return None
        
        # Convert to numpy arrays for easier calculation
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def get_center_point(self, keypoints: np.ndarray, joint1: str, joint2: str) -> Optional[Tuple[float, float]]:
        """Get center point between two joints"""
        p1 = self.get_keypoint(keypoints, joint1)
        p2 = self.get_keypoint(keypoints, joint2)
        
        if not all([p1, p2]):
            return None
        
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2
        
        return (center_x, center_y)
