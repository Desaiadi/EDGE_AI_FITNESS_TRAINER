#!/usr/bin/env python3
"""
EdgeCoach - AI-Powered Fitness Form Coach
Real-time pose estimation and form analysis for squats and planks
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional
import logging

from pose_estimator import PoseEstimator
from exercise_engine import ExerciseEngine
from voice_coach import VoiceCoach
from ui_overlay import UIOverlay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeCoach:
    """Main EdgeCoach application class"""
    
    def __init__(self):
        self.running = False
        self.cap = None
        self.pose_estimator = None
        self.exercise_engine = None
        self.voice_coach = None
        self.ui_overlay = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_history = []
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_thread = None
        
    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
                
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize pose estimator
            self.pose_estimator = PoseEstimator()
            if not self.pose_estimator.initialize():
                logger.error("Failed to initialize pose estimator")
                return False
                
            # Initialize exercise engine
            self.exercise_engine = ExerciseEngine()
            
            # Initialize voice coach
            self.voice_coach = VoiceCoach()
            
            # Initialize UI overlay
            self.ui_overlay = UIOverlay()
            
            logger.info("EdgeCoach initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def start_processing_thread(self):
        """Start the background processing thread"""
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_frames(self):
        """Background thread for processing frames"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    start_time = time.perf_counter()
                    
                    # Process frame
                    keypoints = self.pose_estimator.estimate_pose(frame)
                    
                    if keypoints is not None:
                        # Analyze exercise form
                        analysis = self.exercise_engine.analyze_pose(keypoints)
                        
                        # Generate voice feedback
                        if analysis.get('feedback'):
                            self.voice_coach.speak(analysis['feedback'])
                        
                        # Update UI
                        self.ui_overlay.update_analysis(analysis)
                    
                    # Track latency
                    latency = (time.perf_counter() - start_time) * 1000
                    self.latency_history.append(latency)
                    if len(self.latency_history) > 100:
                        self.latency_history.pop(0)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        self.running = True
        self.start_processing_thread()
        
        logger.info("EdgeCoach started. Press 'q' to quit, 's' for squat, 'p' for plank")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Add frame to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Draw UI overlay
                display_frame = self.ui_overlay.draw_overlay(frame.copy())
                
                # Draw performance metrics
                self._draw_performance_metrics(display_frame)
                
                # Display frame
                cv2.imshow('EdgeCoach - AI Fitness Coach', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.exercise_engine.set_exercise('squat')
                    logger.info("Switched to squat exercise")
                elif key == ord('p'):
                    self.exercise_engine.set_exercise('plank')
                    logger.info("Switched to plank exercise")
                elif key == ord('r'):
                    self.exercise_engine.reset()
                    logger.info("Reset exercise")
                
                # Update FPS counter
                self._update_fps_counter()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def _draw_performance_metrics(self, frame):
        """Draw performance metrics on frame"""
        # FPS
        fps_text = f"FPS: {self.fps_counter:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Latency
        if self.latency_history:
            avg_latency = np.mean(self.latency_history[-10:])
            latency_text = f"Latency: {avg_latency:.1f}ms"
            color = (0, 255, 0) if avg_latency < 100 else (0, 0, 255)
            cv2.putText(frame, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Exercise status
        exercise = self.exercise_engine.get_current_exercise()
        if exercise:
            exercise_text = f"Exercise: {exercise.upper()}"
            cv2.putText(frame, exercise_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.voice_coach:
            self.voice_coach.cleanup()
        
        logger.info("EdgeCoach cleanup completed")

def main():
    """Main entry point"""
    app = EdgeCoach()
    app.run()

if __name__ == "__main__":
    main()
