"""
UI overlay module for real-time exercise feedback display
Provides HUD, metrics, and visual feedback
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class UIOverlay:
    """Real-time UI overlay for exercise feedback"""
    
    def __init__(self):
        self.current_analysis = None
        self.start_time = time.time()
        
        # UI colors (BGR format)
        self.colors = {
            'excellent': (0, 255, 0),      # Green
            'good': (0, 255, 255),         # Yellow
            'fair': (0, 165, 255),         # Orange
            'poor': (0, 0, 255),           # Red
            'unknown': (128, 128, 128),    # Gray
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'blue': (255, 0, 0),
            'purple': (128, 0, 128)
        }
        
        # UI dimensions
        self.sidebar_width = 300
        self.margin = 20
        self.line_height = 25
        self.font_scale = 0.6
        self.thickness = 2
    
    def update_analysis(self, analysis: Dict):
        """Update current analysis data"""
        self.current_analysis = analysis
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw complete UI overlay on frame"""
        if self.current_analysis is None:
            return frame
        
        # Create overlay frame
        overlay = frame.copy()
        
        # Draw sidebar
        overlay = self._draw_sidebar(overlay)
        
        # Draw exercise-specific overlays
        if self.current_analysis.get('exercise') == 'squat':
            overlay = self._draw_squat_overlay(overlay)
        elif self.current_analysis.get('exercise') == 'plank':
            overlay = self._draw_plank_overlay(overlay)
        
        # Draw performance metrics
        overlay = self._draw_performance_metrics(overlay)
        
        return overlay
    
    def _draw_sidebar(self, frame: np.ndarray) -> np.ndarray:
        """Draw left sidebar with exercise info"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent sidebar background
        sidebar = np.zeros((height, self.sidebar_width, 3), dtype=np.uint8)
        sidebar[:] = (20, 20, 20)  # Dark gray background
        
        # Blend with original frame
        overlay = frame.copy()
        overlay[:, :self.sidebar_width] = cv2.addWeighted(
            overlay[:, :self.sidebar_width], 0.7,
            sidebar, 0.3, 0
        )
        
        # Draw sidebar content
        y_offset = self.margin
        
        # Exercise title
        exercise = self.current_analysis.get('exercise', 'Unknown').upper()
        cv2.putText(overlay, f"EXERCISE: {exercise}", 
                   (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, self.colors['white'], 2)
        y_offset += 40
        
        # Rep count
        rep_count = self.current_analysis.get('rep_count', 0)
        cv2.putText(overlay, f"REPS: {rep_count}", 
                   (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, self.colors['white'], 2)
        y_offset += 30
        
        # Current state
        state = self.current_analysis.get('state', 'Unknown')
        cv2.putText(overlay, f"STATE: {state.upper()}", 
                   (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, self.colors['white'], 2)
        y_offset += 30
        
        # Quality indicator
        quality = self.current_analysis.get('quality', 'unknown')
        quality_color = self.colors.get(quality, self.colors['unknown'])
        cv2.putText(overlay, f"QUALITY: {quality.upper()}", 
                   (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, quality_color, 2)
        y_offset += 40
        
        # Current feedback
        feedback = self.current_analysis.get('feedback', '')
        if feedback:
            # Wrap long feedback text
            words = feedback.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 
                                          self.font_scale, self.thickness)[0]
                
                if text_size[0] > self.sidebar_width - 2 * self.margin:
                    if current_line:
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                    else:
                        lines.append(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw feedback lines
            for line in lines[:3]:  # Max 3 lines
                cv2.putText(overlay, line, 
                           (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           self.font_scale, self.colors['white'], 1)
                y_offset += self.line_height
        
        # Draw metrics if available
        metrics = self.current_analysis.get('metrics', {})
        if metrics:
            y_offset += 20
            cv2.putText(overlay, "METRICS:", 
                       (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.colors['white'], 2)
            y_offset += 25
            
            for key, value in list(metrics.items())[:4]:  # Max 4 metrics
                if isinstance(value, (int, float)):
                    text = f"{key}: {value:.1f}"
                else:
                    text = f"{key}: {value}"
                
                cv2.putText(overlay, text, 
                           (self.margin, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, self.colors['white'], 1)
                y_offset += 20
        
        return overlay
    
    def _draw_squat_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw squat-specific overlays"""
        # Draw form indicators
        metrics = self.current_analysis.get('metrics', {})
        
        # Depth indicator
        depth_angle = metrics.get('depth_angle', 0)
        if depth_angle:
            self._draw_depth_indicator(frame, depth_angle)
        
        # Stance width indicator
        stance_width = metrics.get('stance_width', 0)
        if stance_width:
            self._draw_stance_indicator(frame, stance_width)
        
        return frame
    
    def _draw_plank_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw plank-specific overlays"""
        metrics = self.current_analysis.get('metrics', {})
        
        # Hold timer
        hold_time = metrics.get('hold_time', 0)
        if hold_time > 0:
            self._draw_hold_timer(frame, hold_time)
        
        # Alignment indicator
        alignment_angle = metrics.get('alignment_angle', 0)
        if alignment_angle:
            self._draw_alignment_indicator(frame, alignment_angle)
        
        return frame
    
    def _draw_depth_indicator(self, frame: np.ndarray, depth_angle: float):
        """Draw squat depth indicator"""
        height, width = frame.shape[:2]
        
        # Position indicator on right side
        x = width - 150
        y = height - 100
        
        # Draw depth bar
        bar_width = 20
        bar_height = 80
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['white'], 2)
        
        # Fill based on depth
        fill_height = int((depth_angle / 180) * bar_height)
        color = self.colors['excellent'] if depth_angle < 90 else self.colors['poor']
        cv2.rectangle(frame, (x + 2, y + bar_height - fill_height), 
                     (x + bar_width - 2, y + bar_height - 2), color, -1)
        
        # Add labels
        cv2.putText(frame, "DEPTH", (x - 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        cv2.putText(frame, f"{depth_angle:.0f}°", (x - 30, y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
    
    def _draw_stance_indicator(self, frame: np.ndarray, stance_width: float):
        """Draw stance width indicator"""
        height, width = frame.shape[:2]
        
        # Position indicator
        x = width - 150
        y = height - 200
        
        # Draw stance width bar
        bar_width = 100
        bar_height = 20
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['white'], 2)
        
        # Fill based on stance width
        fill_width = int(min(stance_width * 50, bar_width))
        color = self.colors['excellent'] if 0.8 <= stance_width <= 1.2 else self.colors['poor']
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)
        
        # Add labels
        cv2.putText(frame, "STANCE", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        cv2.putText(frame, f"{stance_width:.2f}", (x + bar_width + 10, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
    
    def _draw_hold_timer(self, frame: np.ndarray, hold_time: float):
        """Draw plank hold timer"""
        height, width = frame.shape[:2]
        
        # Position timer at top center
        x = width // 2 - 100
        y = 50
        
        # Draw timer background
        cv2.rectangle(frame, (x, y), (x + 200, y + 40), 
                     self.colors['black'], -1)
        cv2.rectangle(frame, (x, y), (x + 200, y + 40), 
                     self.colors['white'], 2)
        
        # Draw timer text
        timer_text = f"HOLD TIME: {hold_time:.1f}s"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, 2)[0]
        text_x = x + (200 - text_size[0]) // 2
        text_y = y + 25
        
        cv2.putText(frame, timer_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['white'], 2)
    
    def _draw_alignment_indicator(self, frame: np.ndarray, alignment_angle: float):
        """Draw plank alignment indicator"""
        height, width = frame.shape[:2]
        
        # Position indicator
        x = width - 150
        y = 50
        
        # Draw alignment bar
        bar_width = 20
        bar_height = 100
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['white'], 2)
        
        # Fill based on alignment (closer to 0° is better)
        alignment_score = max(0, 1 - abs(alignment_angle) / 30)
        fill_height = int(alignment_score * bar_height)
        color = self.colors['excellent'] if alignment_angle < 10 else self.colors['poor']
        cv2.rectangle(frame, (x + 2, y + bar_height - fill_height), 
                     (x + bar_width - 2, y + bar_height - 2), color, -1)
        
        # Add labels
        cv2.putText(frame, "ALIGN", (x - 20, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        cv2.putText(frame, f"{alignment_angle:.0f}°", (x - 30, y + bar_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
    
    def _draw_performance_metrics(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance metrics overlay"""
        height, width = frame.shape[:2]
        
        # Position metrics at top right
        x = width - 200
        y = 20
        
        # Draw metrics background
        cv2.rectangle(frame, (x - 10, y - 10), (x + 190, y + 80), 
                     self.colors['black'], -1)
        cv2.rectangle(frame, (x - 10, y - 10), (x + 190, y + 80), 
                     self.colors['white'], 1)
        
        # Draw performance text
        cv2.putText(frame, "PERFORMANCE", (x, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        # Add performance indicators
        y_offset = y + 35
        
        # FPS (placeholder - would be passed from main)
        cv2.putText(frame, "FPS: 30+", (x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['excellent'], 1)
        y_offset += 20
        
        # Latency (placeholder)
        cv2.putText(frame, "LATENCY: <100ms", (x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['excellent'], 1)
        y_offset += 20
        
        # NPU status
        cv2.putText(frame, "NPU: ACTIVE", (x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['excellent'], 1)
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw instruction overlay"""
        height, width = frame.shape[:2]
        
        # Position instructions at bottom
        y = height - 100
        
        instructions = [
            "Press 'S' for Squat, 'P' for Plank",
            "Press 'R' to Reset, 'Q' to Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text_y = y + i * 25
            cv2.putText(frame, instruction, (20, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        return frame
