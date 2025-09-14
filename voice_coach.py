"""
Voice coaching module using Windows SAPI
Provides real-time audio feedback for exercise form
"""

import pyttsx3
import threading
import queue
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class VoiceCoach:
    """Real-time voice feedback system"""
    
    def __init__(self):
        self.engine = None
        self.voice_queue = queue.Queue()
        self.speaking = False
        self.last_speech_time = 0
        self.min_speech_interval = 1.0  # Minimum seconds between speeches
        
        # Voice settings
        self.rate = 150  # Words per minute
        self.volume = 0.8  # 0.0 to 1.0
        
        # Feedback messages
        self.feedback_messages = {
            # Squat feedback
            'stance_widen': "Widen your stance",
            'stance_narrow': "Bring your feet closer together",
            'go_deeper': "Go deeper - hips below knees",
            'too_deep': "Don't go too deep",
            'knees_over_toes': "Keep knees over toes",
            'chest_up': "Chest up - don't lean forward",
            'back_straight': "Keep back straight",
            'great_squat': "Great squat!",
            
            # Plank feedback
            'straighten_body': "Straighten your body - head to heels in line",
            'tuck_pelvis': "Tuck your pelvis - don't let hips sag",
            'lower_hips': "Lower your hips - don't pike up",
            'hold_longer': "Hold for {time} more seconds",
            'excellent_plank': "Excellent plank!",
            
            # General feedback
            'position_camera': "Position yourself in front of the camera",
            'get_ready': "Get ready for the exercise",
            'exercise_complete': "Exercise complete! Great work!",
            'reset_position': "Reset your position"
        }
        
        # Initialize voice engine
        self._initialize_voice()
    
    def _initialize_voice(self) -> bool:
        """Initialize the text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice settings
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Try to set a female voice if available
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            # Start the voice processing thread
            self.voice_thread = threading.Thread(target=self._process_voice_queue, daemon=True)
            self.voice_thread.start()
            
            logger.info("Voice coach initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice coach: {e}")
            return False
    
    def speak(self, message: str, priority: str = 'normal'):
        """Add message to voice queue"""
        if not self.engine or not message:
            return
        
        # Check if we should speak (rate limiting)
        current_time = time.time()
        if current_time - self.last_speech_time < self.min_speech_interval:
            return
        
        # Add to queue with priority
        voice_item = {
            'message': message,
            'priority': priority,
            'timestamp': current_time
        }
        
        try:
            self.voice_queue.put(voice_item, timeout=0.1)
        except queue.Full:
            # Queue is full, skip this message
            pass
    
    def speak_feedback(self, feedback_type: str, **kwargs):
        """Speak predefined feedback message"""
        if feedback_type in self.feedback_messages:
            message = self.feedback_messages[feedback_type].format(**kwargs)
            self.speak(message, priority='high')
    
    def speak_custom(self, message: str, priority: str = 'normal'):
        """Speak custom message"""
        self.speak(message, priority)
    
    def _process_voice_queue(self):
        """Process voice queue in background thread"""
        while True:
            try:
                # Get next message from queue
                voice_item = self.voice_queue.get(timeout=1.0)
                
                # Skip if too soon since last speech
                current_time = time.time()
                if current_time - self.last_speech_time < self.min_speech_interval:
                    continue
                
                # Speak the message
                self._speak_message(voice_item['message'])
                
                # Update last speech time
                self.last_speech_time = current_time
                
                # Mark task as done
                self.voice_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Voice processing error: {e}")
    
    def _speak_message(self, message: str):
        """Actually speak the message"""
        try:
            if self.engine and not self.speaking:
                self.speaking = True
                self.engine.say(message)
                self.engine.runAndWait()
                self.speaking = False
        except Exception as e:
            logger.error(f"Speech error: {e}")
            self.speaking = False
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            if self.engine and self.speaking:
                self.engine.stop()
                self.speaking = False
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        if self.engine:
            self.engine.setProperty('rate', rate)
            self.rate = rate
    
    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        if self.engine:
            volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', volume)
            self.volume = volume
    
    def set_min_interval(self, interval: float):
        """Set minimum interval between speeches"""
        self.min_speech_interval = max(0.1, interval)
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.speaking
    
    def get_queue_size(self) -> int:
        """Get current voice queue size"""
        return self.voice_queue.qsize()
    
    def clear_queue(self):
        """Clear the voice queue"""
        while not self.voice_queue.empty():
            try:
                self.voice_queue.get_nowait()
                self.voice_queue.task_done()
            except queue.Empty:
                break
    
    def cleanup(self):
        """Cleanup voice resources"""
        try:
            self.clear_queue()
            if self.engine:
                self.engine.stop()
        except Exception as e:
            logger.error(f"Voice cleanup error: {e}")

class FeedbackManager:
    """Manages feedback timing and prioritization"""
    
    def __init__(self, voice_coach: VoiceCoach):
        self.voice_coach = voice_coach
        self.last_feedback = {}
        self.feedback_cooldowns = {
            'stance': 3.0,
            'depth': 2.0,
            'knees': 2.0,
            'back': 2.0,
            'alignment': 2.0,
            'hips': 2.0,
            'general': 1.0
        }
    
    def provide_feedback(self, feedback_type: str, message: str, category: str = 'general'):
        """Provide feedback with cooldown management"""
        current_time = time.time()
        
        # Check cooldown
        if category in self.last_feedback:
            time_since_last = current_time - self.last_feedback[category]
            cooldown = self.feedback_cooldowns.get(category, 1.0)
            
            if time_since_last < cooldown:
                return False
        
        # Provide feedback
        self.voice_coach.speak(message, priority='high')
        self.last_feedback[category] = current_time
        
        return True
    
    def reset_cooldowns(self):
        """Reset all feedback cooldowns"""
        self.last_feedback.clear()
