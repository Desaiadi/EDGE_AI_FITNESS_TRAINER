"""
Edge AI Trainer - Complete Application for Snapdragon X Elite
A comprehensive AI-powered fitness trainer with local LLM and computer vision capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import threading
import json
import os
import sqlite3
from datetime import datetime, timedelta
import mediapipe as mp
import math
from PIL import Image, ImageTk
import requests
import subprocess
import sys
from pathlib import Path

# Try importing ONNX Runtime for NPU acceleration
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. Install with: pip install onnxruntime")

class LocalLLMManager:
    """Manages local LLM for fitness and nutrition advice"""
    
    def __init__(self):
        self.model_path = None
        self.session = None
        self.knowledge_base = self._load_knowledge_base()
        self.setup_local_model()
    
    def setup_local_model(self):
        """Setup local LLM model for edge inference"""
        # For hackathon demo, we'll use a rule-based system with comprehensive knowledge
        # In production, this would load a quantized local model like Phi-3 or Llama
        print("Setting up local fitness knowledge system...")
        
    def _load_knowledge_base(self):
        """Load comprehensive fitness and nutrition knowledge base"""
        return {
            "exercises": {
                "push_up": {
                    "muscle_groups": ["chest", "shoulders", "triceps", "core"],
                    "calories_per_minute": 7,
                    "difficulty": "beginner",
                    "instructions": [
                        "Start in plank position with hands shoulder-width apart",
                        "Lower body until chest nearly touches floor",
                        "Push back up to starting position",
                        "Keep core engaged throughout movement"
                    ],
                    "form_cues": ["Keep straight line from head to heels", "Don't let hips sag", "Control the descent"]
                },
                "squat": {
                    "muscle_groups": ["quadriceps", "glutes", "hamstrings", "core"],
                    "calories_per_minute": 8,
                    "difficulty": "beginner",
                    "instructions": [
                        "Stand with feet shoulder-width apart",
                        "Lower hips back and down as if sitting in chair",
                        "Keep chest up and knees behind toes",
                        "Return to standing position"
                    ],
                    "form_cues": ["Weight in heels", "Knees track over toes", "Chest up, core tight"]
                },
                "plank": {
                    "muscle_groups": ["core", "shoulders", "back"],
                    "calories_per_minute": 5,
                    "difficulty": "beginner",
                    "instructions": [
                        "Start in forearm plank position",
                        "Keep body straight from head to heels",
                        "Engage core and breathe normally",
                        "Hold position for desired time"
                    ],
                    "form_cues": ["Don't let hips sag", "Keep neck neutral", "Breathe steadily"]
                }
            },
            "nutrition": {
                "macros": {
                    "protein": {"calories_per_gram": 4, "recommended_percentage": 0.25},
                    "carbs": {"calories_per_gram": 4, "recommended_percentage": 0.45},
                    "fats": {"calories_per_gram": 9, "recommended_percentage": 0.30}
                },
                "foods": {
                    "chicken_breast": {"protein": 31, "carbs": 0, "fat": 3.6, "calories": 165},
                    "brown_rice": {"protein": 2.6, "carbs": 23, "fat": 0.9, "calories": 112},
                    "broccoli": {"protein": 2.8, "carbs": 7, "fat": 0.4, "calories": 34},
                    "salmon": {"protein": 25, "carbs": 0, "fat": 12, "calories": 206}
                }
            }
        }
    
    def generate_workout_plan(self, user_profile):
        """Generate personalized workout plan based on user profile"""
        # Calculate BMR and daily caloric needs
        bmr = self._calculate_bmr(user_profile)
        daily_calories = self._calculate_daily_calories(bmr, user_profile['activity_level'])
        
        # Select exercises based on goals
        exercises = self._select_exercises(user_profile['fitness_goal'])
        
        workout_plan = {
            "user_profile": user_profile,
            "daily_calories": daily_calories,
            "bmr": bmr,
            "weekly_schedule": self._create_weekly_schedule(exercises, user_profile),
            "progression": self._create_progression_plan(user_profile)
        }
        
        return workout_plan
    
    def generate_nutrition_plan(self, user_profile):
        """Generate personalized nutrition plan"""
        bmr = self._calculate_bmr(user_profile)
        daily_calories = self._calculate_daily_calories(bmr, user_profile['activity_level'])
        
        # Adjust calories based on goal
        if user_profile['fitness_goal'] == 'weight_loss':
            target_calories = daily_calories * 0.85  # 15% deficit
        elif user_profile['fitness_goal'] == 'muscle_gain':
            target_calories = daily_calories * 1.15  # 15% surplus
        else:
            target_calories = daily_calories
        
        # Calculate macros
        macros = self._calculate_macros(target_calories, user_profile)
        
        # Generate meal plan
        meal_plan = self._create_meal_plan(target_calories, macros, user_profile)
        
        return {
            "target_calories": target_calories,
            "macros": macros,
            "meal_plan": meal_plan,
            "hydration_goal": user_profile['weight'] * 35  # ml per kg
        }
    
    def _calculate_bmr(self, profile):
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
        if profile['gender'] == 'male':
            return 10 * profile['weight'] + 6.25 * profile['height'] - 5 * profile['age'] + 5
        else:
            return 10 * profile['weight'] + 6.25 * profile['height'] - 5 * profile['age'] - 161
    
    def _calculate_daily_calories(self, bmr, activity_level):
        """Calculate daily caloric needs based on activity level"""
        multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        return bmr * multipliers.get(activity_level, 1.375)
    
    def _select_exercises(self, goal):
        """Select appropriate exercises based on fitness goal"""
        if goal == 'weight_loss':
            return ['push_up', 'squat', 'plank']  # High intensity, compound movements
        elif goal == 'muscle_gain':
            return ['push_up', 'squat']  # Strength-focused
        else:
            return ['push_up', 'squat', 'plank']  # General fitness
    
    def _create_weekly_schedule(self, exercises, profile):
        """Create a weekly workout schedule"""
        schedule = {}
        for day in ['Monday', 'Wednesday', 'Friday']:
            schedule[day] = {
                'exercises': exercises,
                'sets': 3 if profile['fitness_level'] != 'beginner' else 2,
                'reps': self._calculate_reps(profile['fitness_level']),
                'rest_between_sets': 60
            }
        return schedule
    
    def _calculate_reps(self, fitness_level):
        """Calculate appropriate reps based on fitness level"""
        reps_map = {
            'beginner': {'push_up': 8, 'squat': 12, 'plank': 30},
            'intermediate': {'push_up': 12, 'squat': 15, 'plank': 45},
            'advanced': {'push_up': 15, 'squat': 20, 'plank': 60}
        }
        return reps_map.get(fitness_level, reps_map['beginner'])
    
    def _create_progression_plan(self, profile):
        """Create progression plan for advancing difficulty"""
        return {
            "week_1_2": "Focus on form and consistency",
            "week_3_4": "Increase reps by 2-3 per exercise",
            "week_5_6": "Add additional set or increase hold time",
            "week_7_8": "Introduce exercise variations"
        }
    
    def _calculate_macros(self, calories, profile):
        """Calculate macro distribution"""
        macros = self.knowledge_base['nutrition']['macros']
        return {
            'protein': int(calories * macros['protein']['recommended_percentage'] / macros['protein']['calories_per_gram']),
            'carbs': int(calories * macros['carbs']['recommended_percentage'] / macros['carbs']['calories_per_gram']),
            'fats': int(calories * macros['fats']['recommended_percentage'] / macros['fats']['calories_per_gram'])
        }
    
    def _create_meal_plan(self, calories, macros, profile):
        """Create sample meal plan"""
        # Simplified meal planning for demo
        return {
            "breakfast": {
                "foods": ["oatmeal", "banana", "almonds"],
                "calories": int(calories * 0.25)
            },
            "lunch": {
                "foods": ["chicken_breast", "brown_rice", "broccoli"],
                "calories": int(calories * 0.35)
            },
            "dinner": {
                "foods": ["salmon", "quinoa", "asparagus"],
                "calories": int(calories * 0.30)
            },
            "snacks": {
                "foods": ["greek_yogurt", "berries"],
                "calories": int(calories * 0.10)
            }
        }

class ExerciseTracker:
    """Computer vision-based exercise tracking and form correction"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise counters and states
        self.exercise_counts = {'push_up': 0, 'squat': 0, 'plank_time': 0}
        self.exercise_state = {'push_up': 'up', 'squat': 'up', 'plank_start_time': None}
        self.current_exercise = None
        self.form_feedback = []
        
        # Angle calculation thresholds
        self.push_up_thresholds = {'down': 90, 'up': 160}
        self.squat_thresholds = {'down': 90, 'up': 160}
        self.plank_thresholds = {'good_form': 15}  # degrees deviation from straight
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def track_push_ups(self, landmarks):
        """Track push-up repetitions and form"""
        # Get relevant landmarks
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate elbow angle
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Count repetitions
        if elbow_angle < self.push_up_thresholds['down'] and self.exercise_state['push_up'] == 'up':
            self.exercise_state['push_up'] = 'down'
        
        if elbow_angle > self.push_up_thresholds['up'] and self.exercise_state['push_up'] == 'down':
            self.exercise_counts['push_up'] += 1
            self.exercise_state['push_up'] = 'up'
        
        # Form feedback
        self.form_feedback = []
        if elbow_angle < 70:
            self.form_feedback.append("Go deeper - chest closer to floor")
        elif elbow_angle > 170:
            self.form_feedback.append("Good range of motion!")
        
        # Check body alignment
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        body_angle = self.calculate_angle(ankle, hip, shoulder)
        if abs(body_angle - 180) > 15:
            self.form_feedback.append("Keep body straight - engage core")
            
        return elbow_angle
    
    def track_squats(self, landmarks):
        """Track squat repetitions and form"""
        # Get relevant landmarks
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate knee angle
        knee_angle = self.calculate_angle(hip, knee, ankle)
        
        # Count repetitions
        if knee_angle < self.squat_thresholds['down'] and self.exercise_state['squat'] == 'up':
            self.exercise_state['squat'] = 'down'
        
        if knee_angle > self.squat_thresholds['up'] and self.exercise_state['squat'] == 'down':
            self.exercise_counts['squat'] += 1
            self.exercise_state['squat'] = 'up'
        
        # Form feedback
        self.form_feedback = []
        if knee_angle > 100:
            self.form_feedback.append("Go lower - thighs parallel to floor")
        elif knee_angle < 70:
            self.form_feedback.append("Great depth!")
        
        # Check knee alignment
        if abs(hip[0] - knee[0]) > 0.1:  # Normalized coordinates
            self.form_feedback.append("Keep knees in line with toes")
            
        return knee_angle
    
    def track_plank(self, landmarks):
        """Track plank form and duration"""
        if self.exercise_state['plank_start_time'] is None:
            self.exercise_state['plank_start_time'] = datetime.now()
        
        # Calculate plank duration
        current_time = datetime.now()
        duration = (current_time - self.exercise_state['plank_start_time']).total_seconds()
        self.exercise_counts['plank_time'] = int(duration)
        
        # Check form
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        body_angle = self.calculate_angle(ankle, hip, shoulder)
        deviation = abs(180 - body_angle)
        
        self.form_feedback = []
        if deviation < self.plank_thresholds['good_form']:
            self.form_feedback.append("Perfect form! Keep it up!")
        elif deviation < 25:
            self.form_feedback.append("Good form - small adjustment needed")
        else:
            if hip[1] < shoulder[1]:  # Hip too high
                self.form_feedback.append("Lower your hips")
            else:  # Hip too low
                self.form_feedback.append("Raise your hips - engage core")
        
        return deviation
    
    def process_frame(self, frame, exercise_type):
        """Process video frame for exercise tracking"""
        self.current_exercise = exercise_type
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )
            
            # Track specific exercise
            if exercise_type == 'push_up':
                angle = self.track_push_ups(results.pose_landmarks.landmark)
                cv2.putText(frame, f'Elbow Angle: {int(angle)}', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif exercise_type == 'squat':
                angle = self.track_squats(results.pose_landmarks.landmark)
                cv2.putText(frame, f'Knee Angle: {int(angle)}', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            elif exercise_type == 'plank':
                deviation = self.track_plank(results.pose_landmarks.landmark)
                cv2.putText(frame, f'Body Alignment: {int(deviation)}°', (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display exercise info
        cv2.putText(frame, f'Exercise: {exercise_type.title()}', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Count: {self.exercise_counts.get(exercise_type, 0)}', (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display form feedback
        for i, feedback in enumerate(self.form_feedback):
            cv2.putText(frame, feedback, (50, 200 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def reset_exercise(self):
        """Reset exercise counters and state"""
        self.exercise_counts = {'push_up': 0, 'squat': 0, 'plank_time': 0}
        self.exercise_state = {'push_up': 'up', 'squat': 'up', 'plank_start_time': None}
        self.form_feedback = []

class DatabaseManager:
    """Manages local SQLite database for user data and workout history"""
    
    def __init__(self, db_path="fitness_trainer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                weight REAL,
                height REAL,
                gender TEXT,
                activity_level TEXT,
                fitness_goal TEXT,
                fitness_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Workout history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workout_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                exercise_type TEXT,
                reps INTEGER,
                duration INTEGER,
                calories_burned REAL,
                workout_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (id)
            )
        ''')
        
        # Nutrition logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nutrition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                meal_type TEXT,
                foods TEXT,
                calories REAL,
                protein REAL,
                carbs REAL,
                fats REAL,
                log_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user_profile(self, profile_data):
        """Create new user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_profiles 
            (name, age, weight, height, gender, activity_level, fitness_goal, fitness_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile_data['name'], profile_data['age'], profile_data['weight'],
            profile_data['height'], profile_data['gender'], profile_data['activity_level'],
            profile_data['fitness_goal'], profile_data['fitness_level']
        ))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    def get_user_profiles(self):
        """Get all user profiles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_profiles ORDER BY created_at DESC')
        profiles = cursor.fetchall()
        
        conn.close()
        return profiles
    
    def log_workout(self, user_id, exercise_data):
        """Log workout session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO workout_history 
            (user_id, exercise_type, reps, duration, calories_burned)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id, exercise_data['exercise_type'], exercise_data['reps'],
            exercise_data['duration'], exercise_data['calories_burned']
        ))
        
        conn.commit()
        conn.close()
    
    def get_workout_history(self, user_id, days=30):
        """Get workout history for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM workout_history 
            WHERE user_id = ? AND workout_date >= datetime('now', '-{} days')
            ORDER BY workout_date DESC
        '''.format(days), (user_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        return history

class AITrainerApp:
    """Main application class for the Edge AI Trainer"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Edge AI Trainer - Snapdragon X Elite")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        
        # Initialize components
        self.llm_manager = LocalLLMManager()
        self.exercise_tracker = ExerciseTracker()
        self.db_manager = DatabaseManager()
        
        # Video capture
        self.cap = None
        self.camera_running = False
        
        # Current user and workout data
        self.current_user = None
        self.current_workout_plan = None
        self.current_nutrition_plan = None
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        
    def setup_styles(self):
        """Configure UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles for dark theme
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background="#1a1a1a", foreground="white")
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), background="#1a1a1a", foreground="white")
        style.configure('Info.TLabel', font=('Arial', 10), background="#1a1a1a", foreground="white")
        style.configure('Custom.TButton', font=('Arial', 10, 'bold'))
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_profile_tab()
        self.setup_planner_tab()
        self.setup_workout_tab()
        self.setup_history_tab()
        
    def setup_profile_tab(self):
        """Setup user profile creation tab"""
        profile_frame = ttk.Frame(self.notebook)
        self.notebook.add(profile_frame, text="User Profile")
        
        # Title
        title_label = ttk.Label(profile_frame, text="Create Your Fitness Profile", 
                               style="Title.TLabel")
        title_label.pack(pady=20)
        
        # Profile form
        form_frame = ttk.Frame(profile_frame)
        form_frame.pack(pady=20)
        
        # Form fields
        self.profile_vars = {}
        fields = [
            ("Name:", "name", "text"),
            ("Age:", "age", "number"),
            ("Weight (kg):", "weight", "number"),
            ("Height (cm):", "height", "number"),
            ("Gender:", "gender", "combo"),
            ("Activity Level:", "activity_level", "combo"),
            ("Fitness Goal:", "fitness_goal", "combo"),
            ("Fitness Level:", "fitness_level", "combo")
        ]
        
        combo_values = {
            "gender": ["male", "female"],
            "activity_level": ["sedentary", "light", "moderate", "active", "very_active"],
            "fitness_goal": ["weight_loss", "muscle_gain", "endurance", "general_fitness"],
            "fitness_level": ["beginner", "intermediate", "advanced"]
        }
        
        for i, (label_text, var_name, field_type) in enumerate(fields):
            # Label
            label = ttk.Label(form_frame, text=label_text, style="Info.TLabel")
            label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
            
            # Entry/Combobox
            if field_type == "combo":
                var = tk.StringVar()
                widget = ttk.Combobox(form_frame, textvariable=var, 
                                     values=combo_values[var_name], state="readonly")
            else:
                var = tk.StringVar() if field_type == "text" else tk.DoubleVar()
                widget = ttk.Entry(form_frame, textvariable=var)
            
            widget.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            self.profile_vars[var_name] = var
        
        form_frame.columnconfigure(1, weight=1)
        
        # Create profile button
        create_btn = ttk.Button(form_frame, text="Create Profile", 
                               command=self.create_profile, style="Custom.TButton")
        create_btn.grid(row=len(fields), column=0, columnspan=2, pady=20)
        
        # User selection
        select_frame = ttk.Frame(profile_frame)
        select_frame.pack(pady=20, fill="x")
        
        ttk.Label(select_frame, text="Select User:", style="Subtitle.TLabel").pack()
        
        self.user_combo = ttk.Combobox(select_frame, state="readonly")
        self.user_combo.pack(pady=10)
        self.user_combo.bind('<<ComboboxSelected>>', self.on_user_selected)
        
        self.refresh_users()
        
    def setup_planner_tab(self):
        """Setup workout and nutrition planner tab"""
        planner_frame = ttk.Frame(self.notebook)
        self.notebook.add(planner_frame, text="AI Planner")
        
        # Main container with scrollable text
        main_frame = ttk.Frame(planner_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="AI Fitness & Nutrition Planner", 
                               style="Title.TLabel")
        title_label.pack(pady=10)
        
        # Generate plans button
        generate_frame = ttk.Frame(main_frame)
        generate_frame.pack(pady=10)
        
        ttk.Button(generate_frame, text="Generate Workout Plan", 
                  command=self.generate_workout_plan, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(generate_frame, text="Generate Nutrition Plan", 
                  command=self.generate_nutrition_plan, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.planner_text = tk.Text(main_frame, height=25, bg="#2a2a2a", fg="white", 
                                   font=('Consolas', 11), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.planner_text.yview)
        self.planner_text.configure(yscrollcommand=scrollbar.set)
        
        self.planner_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_workout_tab(self):
        """Setup live workout tracking tab"""
        workout_frame = ttk.Frame(self.notebook)
        self.notebook.add(workout_frame, text="Live Workout")
        
        # Title
        title_label = ttk.Label(workout_frame, text="Live Workout Tracking", 
                               style="Title.TLabel")
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.Frame(workout_frame)
        control_frame.pack(pady=10)
        
        # Exercise selection
        ttk.Label(control_frame, text="Exercise:", style="Info.TLabel").pack(side=tk.LEFT, padx=5)
        self.exercise_var = tk.StringVar(value="push_up")
        exercise_combo = ttk.Combobox(control_frame, textvariable=self.exercise_var,
                                     values=["push_up", "squat", "plank"], state="readonly")
        exercise_combo.pack(side=tk.LEFT, padx=5)
        
        # Camera controls
        ttk.Button(control_frame, text="Start Camera", 
                  command=self.start_camera, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Camera", 
                  command=self.stop_camera, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset Count", 
                  command=self.reset_exercise_count, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        
        # Video display frame
        video_frame = ttk.Frame(workout_frame)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera display
        self.video_label = tk.Label(video_frame, bg="black", text="Camera Off", 
                                   fg="white", font=('Arial', 20))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Stats panel
        stats_frame = ttk.Frame(workout_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_labels = {}
        stats_info = [
            ("Exercise:", "current_exercise"),
            ("Count:", "count"), 
            ("Duration:", "duration"),
            ("Calories:", "calories"),
            ("Form Feedback:", "feedback")
        ]
        
        for i, (label_text, key) in enumerate(stats_info):
            frame = ttk.Frame(stats_frame)
            frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            ttk.Label(frame, text=label_text, style="Info.TLabel").pack()
            self.stats_labels[key] = ttk.Label(frame, text="--", style="Subtitle.TLabel")
            self.stats_labels[key].pack()
        
    def setup_history_tab(self):
        """Setup workout history and progress tracking tab"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="Progress & History")
        
        # Title
        title_label = ttk.Label(history_frame, text="Workout History & Progress", 
                               style="Title.TLabel")
        title_label.pack(pady=10)
        
        # History display
        self.history_tree = ttk.Treeview(history_frame, columns=("Date", "Exercise", "Reps", "Duration", "Calories"),
                                        show="headings", height=15)
        
        # Configure columns
        self.history_tree.heading("Date", text="Date")
        self.history_tree.heading("Exercise", text="Exercise")
        self.history_tree.heading("Reps", text="Reps/Time")
        self.history_tree.heading("Duration", text="Duration (min)")
        self.history_tree.heading("Calories", text="Calories")
        
        self.history_tree.column("Date", width=120)
        self.history_tree.column("Exercise", width=100)
        self.history_tree.column("Reps", width=80)
        self.history_tree.column("Duration", width=100)
        self.history_tree.column("Calories", width=80)
        
        # Scrollbar for history
        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, 
                                         command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0), pady=10)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10), pady=10)
        
        # Refresh button
        ttk.Button(history_frame, text="Refresh History", 
                  command=self.refresh_history, style="Custom.TButton").pack(pady=10)
    
    def create_profile(self):
        """Create new user profile"""
        try:
            # Validate inputs
            profile_data = {}
            for key, var in self.profile_vars.items():
                value = var.get()
                if not value:
                    messagebox.showerror("Error", f"Please fill in {key.replace('_', ' ').title()}")
                    return
                profile_data[key] = value
            
            # Create profile in database
            user_id = self.db_manager.create_user_profile(profile_data)
            
            messagebox.showinfo("Success", f"Profile created successfully! User ID: {user_id}")
            self.refresh_users()
            
            # Clear form
            for var in self.profile_vars.values():
                var.set("")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create profile: {str(e)}")
    
    def refresh_users(self):
        """Refresh user dropdown"""
        try:
            profiles = self.db_manager.get_user_profiles()
            user_list = [f"{profile[1]} (ID: {profile[0]})" for profile in profiles]
            self.user_combo['values'] = user_list
            
        except Exception as e:
            print(f"Error refreshing users: {e}")
    
    def on_user_selected(self, event):
        """Handle user selection"""
        selection = self.user_combo.get()
        if selection:
            user_id = int(selection.split("ID: ")[1].split(")")[0])
            profiles = self.db_manager.get_user_profiles()
            
            for profile in profiles:
                if profile[0] == user_id:
                    self.current_user = {
                        'id': profile[0],
                        'name': profile[1],
                        'age': profile[2],
                        'weight': profile[3],
                        'height': profile[4],
                        'gender': profile[5],
                        'activity_level': profile[6],
                        'fitness_goal': profile[7],
                        'fitness_level': profile[8]
                    }
                    break
            
            messagebox.showinfo("User Selected", f"Selected user: {self.current_user['name']}")
            self.refresh_history()
    
    def generate_workout_plan(self):
        """Generate personalized workout plan"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user profile first")
            return
        
        try:
            self.current_workout_plan = self.llm_manager.generate_workout_plan(self.current_user)
            
            # Display plan
            plan_text = self.format_workout_plan(self.current_workout_plan)
            self.planner_text.delete(1.0, tk.END)
            self.planner_text.insert(tk.END, plan_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate workout plan: {str(e)}")
    
    def generate_nutrition_plan(self):
        """Generate personalized nutrition plan"""
        if not self.current_user:
            messagebox.showerror("Error", "Please select a user profile first")
            return
        
        try:
            self.current_nutrition_plan = self.llm_manager.generate_nutrition_plan(self.current_user)
            
            # Display plan
            plan_text = self.format_nutrition_plan(self.current_nutrition_plan)
            self.planner_text.delete(1.0, tk.END)
            self.planner_text.insert(tk.END, plan_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate nutrition plan: {str(e)}")
    
    def format_workout_plan(self, plan):
        """Format workout plan for display"""
        text = f"""
═══════════════════════════════════════════════
🏋️  PERSONALIZED WORKOUT PLAN
═══════════════════════════════════════════════

👤 USER PROFILE
Name: {self.current_user['name']}
Goal: {self.current_user['fitness_goal'].replace('_', ' ').title()}
Level: {self.current_user['fitness_level'].title()}

📊 METABOLIC DATA
BMR: {plan['bmr']:.0f} calories/day
Daily Calories: {plan['daily_calories']:.0f} calories/day

📅 WEEKLY SCHEDULE
"""
        
        for day, workout in plan['weekly_schedule'].items():
            text += f"\n🗓️  {day}:\n"
            text += f"   Sets: {workout['sets']}\n"
            text += f"   Rest: {workout['rest_between_sets']}s between sets\n"
            text += f"   Exercises:\n"
            
            for exercise in workout['exercises']:
                reps = workout['reps'][exercise]
                exercise_info = self.llm_manager.knowledge_base['exercises'][exercise]
                calories = exercise_info['calories_per_minute'] * 3  # Approximate
                
                text += f"   • {exercise.replace('_', ' ').title()}: {reps} reps/seconds\n"
                text += f"     Target: {', '.join(exercise_info['muscle_groups'])}\n"
                text += f"     Calories: ~{calories}/set\n"
        
        text += f"\n📈 PROGRESSION PLAN\n"
        for week, instruction in plan['progression'].items():
            text += f"   {week.replace('_', ' ').title()}: {instruction}\n"
        
        text += f"\n💡 KEY TIPS\n"
        text += f"   • Focus on proper form over speed\n"
        text += f"   • Rest 48 hours between intense sessions\n"
        text += f"   • Stay hydrated throughout workouts\n"
        text += f"   • Listen to your body and adjust as needed\n"
        
        return text
    
    def format_nutrition_plan(self, plan):
        """Format nutrition plan for display"""
        text = f"""
═══════════════════════════════════════════════
🍽️  PERSONALIZED NUTRITION PLAN
═══════════════════════════════════════════════

👤 USER PROFILE
Name: {self.current_user['name']}
Goal: {self.current_user['fitness_goal'].replace('_', ' ').title()}

📊 DAILY TARGETS
Calories: {plan['target_calories']:.0f} cal/day
Protein: {plan['macros']['protein']}g ({plan['macros']['protein']*4:.0f} cal)
Carbs: {plan['macros']['carbs']}g ({plan['macros']['carbs']*4:.0f} cal)  
Fats: {plan['macros']['fats']}g ({plan['macros']['fats']*9:.0f} cal)
Water: {plan['hydration_goal']:.0f}ml/day

🍽️  DAILY MEAL PLAN

🌅 Breakfast ({plan['meal_plan']['breakfast']['calories']} cal):
   • {', '.join(plan['meal_plan']['breakfast']['foods'])}
   
☀️  Lunch ({plan['meal_plan']['lunch']['calories']} cal):
   • {', '.join(plan['meal_plan']['lunch']['foods'])}
   
🌙 Dinner ({plan['meal_plan']['dinner']['calories']} cal):
   • {', '.join(plan['meal_plan']['dinner']['foods'])}
   
🥜 Snacks ({plan['meal_plan']['snacks']['calories']} cal):
   • {', '.join(plan['meal_plan']['snacks']['foods'])}

💡 NUTRITION TIPS
   • Eat protein with every meal for muscle recovery
   • Time carbs around workouts for energy
   • Include healthy fats for hormone production
   • Drink water consistently throughout the day
   • Focus on whole, minimally processed foods
   
📝 MEAL PREP SUGGESTIONS
   • Cook proteins in bulk on Sundays
   • Pre-cut vegetables for easy access
   • Prepare overnight oats for quick breakfasts
   • Keep healthy snacks readily available
"""
        return text
    
    def start_camera(self):
        """Start camera for workout tracking"""
        if not self.camera_running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Cannot access camera")
                    return
                
                self.camera_running = True
                self.exercise_tracker.reset_exercise()
                self.update_camera()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera and save workout"""
        if self.camera_running:
            self.camera_running = False
            if self.cap:
                self.cap.release()
            
            self.video_label.configure(image="", text="Camera Off")
            
            # Save workout if user is selected
            if self.current_user:
                self.save_workout_session()
    
    def update_camera(self):
        """Update camera feed and exercise tracking"""
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for exercise tracking
                exercise_type = self.exercise_var.get()
                processed_frame = self.exercise_tracker.process_frame(frame, exercise_type)
                
                # Convert to PhotoImage for tkinter
                processed_frame = cv2.resize(processed_frame, (640, 480))
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(processed_frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=imgtk, text="")
                self.video_label.imgtk = imgtk  # Keep a reference
                
                # Update stats
                self.update_workout_stats(exercise_type)
            
            # Schedule next update
            self.root.after(33, self.update_camera)  # ~30 FPS
    
    def update_workout_stats(self, exercise_type):
        """Update workout statistics display"""
        counts = self.exercise_tracker.exercise_counts
        feedback = self.exercise_tracker.form_feedback
        
        self.stats_labels['current_exercise'].config(text=exercise_type.replace('_', ' ').title())
        
        if exercise_type == 'plank':
            self.stats_labels['count'].config(text=f"{counts['plank_time']}s")
            duration = counts['plank_time'] / 60  # Convert to minutes
        else:
            self.stats_labels['count'].config(text=str(counts.get(exercise_type, 0)))
            duration = 1  # Approximate duration for rep-based exercises
        
        self.stats_labels['duration'].config(text=f"{duration:.1f} min")
        
        # Estimate calories (simplified calculation)
        exercise_info = self.llm_manager.knowledge_base['exercises'].get(exercise_type, {})
        calories_per_min = exercise_info.get('calories_per_minute', 5)
        total_calories = duration * calories_per_min
        self.stats_labels['calories'].config(text=f"{total_calories:.0f}")
        
        # Form feedback
        feedback_text = feedback[0] if feedback else "Good form!"
        self.stats_labels['feedback'].config(text=feedback_text[:30])
    
    def reset_exercise_count(self):
        """Reset exercise counters"""
        self.exercise_tracker.reset_exercise()
        self.stats_labels['count'].config(text="0")
        self.stats_labels['duration'].config(text="0.0 min")
        self.stats_labels['calories'].config(text="0")
        self.stats_labels['feedback'].config(text="--")
    
    def save_workout_session(self):
        """Save completed workout session"""
        if not self.current_user:
            return
        
        try:
            exercise_type = self.exercise_var.get()
            counts = self.exercise_tracker.exercise_counts
            
            if exercise_type == 'plank':
                reps = counts['plank_time']  # Duration in seconds
                duration = counts['plank_time'] / 60  # Convert to minutes
            else:
                reps = counts.get(exercise_type, 0)
                duration = 5  # Approximate duration for rep-based exercises
            
            # Calculate calories burned
            exercise_info = self.llm_manager.knowledge_base['exercises'].get(exercise_type, {})
            calories_per_min = exercise_info.get('calories_per_minute', 5)
            calories_burned = duration * calories_per_min
            
            # Save to database
            workout_data = {
                'exercise_type': exercise_type,
                'reps': reps,
                'duration': int(duration),
                'calories_burned': calories_burned
            }
            
            self.db_manager.log_workout(self.current_user['id'], workout_data)
            messagebox.showinfo("Workout Saved", 
                              f"Great job! Workout saved:\n{exercise_type.replace('_', ' ').title()}\n"
                              f"Reps/Time: {reps}\nCalories: {calories_burned:.0f}")
            
            self.refresh_history()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save workout: {str(e)}")
    
    def refresh_history(self):
        """Refresh workout history display"""
        if not self.current_user:
            return
        
        try:
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Get workout history
            history = self.db_manager.get_workout_history(self.current_user['id'])
            
            for workout in history:
                # Format date
                date_str = workout[6][:10]  # Get date part only
                exercise = workout[2].replace('_', ' ').title()
                reps = f"{workout[3]}{'s' if workout[2] == 'plank' else ''}"
                duration = f"{workout[4]}"
                calories = f"{workout[5]:.0f}"
                
                self.history_tree.insert("", "end", values=(date_str, exercise, reps, duration, calories))
                
        except Exception as e:
            print(f"Error refreshing history: {e}")

def main():
    """Main function to run the application"""
    # Check for required dependencies
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        return
    
    # Create and run application
    root = tk.Tk()
    app = AITrainerApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if hasattr(app, 'cap') and app.cap:
            app.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()