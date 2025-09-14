#!/usr/bin/env python3
"""
EdgeCoach Show Demo - Demonstrates real-time functionality
Shows how EdgeCoach works with simulated real-time data
"""

import time
import random
import math
import os

def clear_screen():
    """Clear the screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_header():
    """Show demo header"""
    print("=" * 80)
    print("🎯 EDGECOACH - AI FITNESS COACH (Real-time Demo)")
    print("=" * 80)
    print("This demo shows how EdgeCoach works in real-time:")
    print("• Pose detection and tracking")
    print("• Exercise form analysis")
    print("• Voice feedback system")
    print("• Performance monitoring")
    print("=" * 80)

def simulate_pose_detection(exercise_type, frame_count):
    """Simulate realistic pose detection"""
    # Simulate different poses based on exercise and frame
    if exercise_type == 'squat':
        return simulate_squat_pose(frame_count)
    elif exercise_type == 'plank':
        return simulate_plank_pose(frame_count)
    else:
        return simulate_standing_pose()

def simulate_squat_pose(frame_count):
    """Simulate squat pose sequence"""
    # Create a realistic squat sequence
    phase = (frame_count // 10) % 8  # 8 phases, 10 frames each
    
    poses = [
        {'name': 'Setup', 'depth': 180, 'stance': 1.0, 'knees': 0.1, 'description': 'Standing ready'},
        {'name': 'Descent', 'depth': 150, 'stance': 1.0, 'knees': 0.2, 'description': 'Going down'},
        {'name': 'Descent', 'depth': 120, 'stance': 1.0, 'knees': 0.3, 'description': 'Lowering'},
        {'name': 'Descent', 'depth': 100, 'stance': 1.0, 'knees': 0.4, 'description': 'Almost there'},
        {'name': 'Bottom', 'depth': 85, 'stance': 1.0, 'knees': 0.2, 'description': 'Deep squat!'},
        {'name': 'Ascent', 'depth': 100, 'stance': 1.0, 'knees': 0.3, 'description': 'Coming up'},
        {'name': 'Ascent', 'depth': 130, 'stance': 1.0, 'knees': 0.2, 'description': 'Rising'},
        {'name': 'Lockout', 'depth': 170, 'stance': 1.0, 'knees': 0.1, 'description': 'Standing tall!'}
    ]
    
    return poses[phase]

def simulate_plank_pose(frame_count):
    """Simulate plank pose sequence"""
    # Create a realistic plank sequence
    phase = (frame_count // 15) % 6  # 6 phases, 15 frames each
    
    poses = [
        {'name': 'Setup', 'alignment': 5, 'hip_sag': 0.02, 'description': 'Getting into position'},
        {'name': 'Holding', 'alignment': 8, 'hip_sag': 0.05, 'description': 'Good form'},
        {'name': 'Holding', 'alignment': 6, 'hip_sag': 0.03, 'description': 'Perfect alignment'},
        {'name': 'Hip Sag', 'alignment': 15, 'hip_sag': 0.12, 'description': 'Hips sagging'},
        {'name': 'Holding', 'alignment': 7, 'hip_sag': 0.04, 'description': 'Corrected form'},
        {'name': 'Holding', 'alignment': 5, 'hip_sag': 0.02, 'description': 'Excellent hold'}
    ]
    
    return poses[phase]

def simulate_standing_pose():
    """Simulate standing pose"""
    return {'name': 'Standing', 'depth': 180, 'stance': 1.0, 'knees': 0.1, 'description': 'Ready position'}

def analyze_squat_form(pose_data, rep_count):
    """Analyze squat form and generate feedback"""
    depth = pose_data['depth']
    stance = pose_data['stance']
    knees = pose_data['knees']
    
    feedback = None
    quality = 'good'
    
    if pose_data['name'] == 'Setup':
        feedback = "Get ready! Stand with feet shoulder-width apart"
        quality = 'good'
    elif pose_data['name'] == 'Descent':
        if depth > 120:
            feedback = "Go deeper! Hips should go below knees"
            quality = 'fair'
        else:
            feedback = "Good depth! Keep going down slowly"
            quality = 'good'
    elif pose_data['name'] == 'Bottom':
        if depth < 90:
            feedback = "Perfect depth! Now drive up through your heels"
            quality = 'excellent'
        else:
            feedback = "Almost there! Go a bit deeper"
            quality = 'fair'
    elif pose_data['name'] == 'Ascent':
        feedback = "Great! Drive up through your heels, chest up"
        quality = 'good'
    elif pose_data['name'] == 'Lockout':
        rep_count += 1
        feedback = f"Excellent rep! Total: {rep_count} squats"
        quality = 'excellent'
    
    return {
        'feedback': feedback,
        'quality': quality,
        'rep_count': rep_count,
        'metrics': {
            'depth_angle': depth,
            'stance_width': stance,
            'knee_over_toe': knees
        }
    }

def analyze_plank_form(pose_data, hold_time):
    """Analyze plank form and generate feedback"""
    alignment = pose_data['alignment']
    hip_sag = pose_data['hip_sag']
    
    feedback = None
    quality = 'good'
    
    if pose_data['name'] == 'Setup':
        feedback = "Get into plank position! Hands under shoulders"
        quality = 'good'
    elif pose_data['name'] == 'Holding':
        if alignment < 10:
            feedback = f"Perfect plank! Hold for {max(0, 5 - hold_time):.1f} more seconds"
            quality = 'excellent'
        else:
            feedback = "Good hold! Keep your body straight"
            quality = 'good'
    elif pose_data['name'] == 'Hip Sag':
        feedback = "Hips are sagging! Tuck your pelvis and engage your core"
        quality = 'fair'
    
    return {
        'feedback': feedback,
        'quality': quality,
        'rep_count': int(hold_time // 10),
        'metrics': {
            'alignment_angle': alignment,
            'hip_sag': hip_sag,
            'hold_time': hold_time
        }
    }

def show_pose_overlay(pose_data, analysis, frame_count, fps, latency):
    """Show pose overlay with skeleton visualization"""
    print("🎯 POSE DETECTION OVERLAY")
    print("-" * 40)
    
    # Simulate skeleton drawing
    print("    👤 Skeleton Detection:")
    print("       Head: ●")
    print("       Shoulders: ●─────●")
    print("       Arms: ●─●─●")
    print("       Torso: │")
    print("       Hips: ●─────●")
    print("       Legs: ●─●─●")
    print()
    
    # Show keypoints
    print("    📍 Key Points Detected:")
    keypoints = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]
    
    for i, point in enumerate(keypoints[:8]):  # Show first 8 keypoints
        confidence = random.uniform(0.7, 0.95)
        status = "✓" if confidence > 0.8 else "⚠"
        print(f"       {point}: {status} ({confidence:.2f})")
    
    print()

def show_exercise_analysis(exercise, analysis, frame_count):
    """Show exercise analysis"""
    print("🏃‍♂️ EXERCISE ANALYSIS")
    print("-" * 40)
    print(f"Exercise: {exercise.upper()}")
    print(f"State: {analysis.get('state', 'Unknown').upper()}")
    print(f"Reps: {analysis.get('rep_count', 0)}")
    print(f"Quality: {analysis.get('quality', 'Unknown').upper()}")
    print()
    
    # Show metrics
    metrics = analysis.get('metrics', {})
    if metrics:
        print("📊 Form Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key.replace('_', ' ').title()}: {value:.1f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")
        print()
    
    # Show feedback
    feedback = analysis.get('feedback', '')
    if feedback:
        print(f"💬 Feedback: {feedback}")
        print()

def show_performance_metrics(fps, latency, frame_count):
    """Show performance metrics"""
    print("⚡ PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Frame Rate: {fps:.1f} FPS")
    print(f"Latency: {latency:.1f}ms")
    print(f"Frame Count: {frame_count}")
    print(f"NPU Status: DirectML Active")
    print(f"Memory Usage: <2GB")
    print()

def show_controls():
    """Show control instructions"""
    print("🎮 CONTROLS")
    print("-" * 40)
    print("S - Switch to Squat exercise")
    print("P - Switch to Plank exercise")
    print("R - Reset exercise")
    print("Q - Quit demo")
    print()

def main():
    """Main demo function"""
    clear_screen()
    show_header()
    
    print("Press Enter to start the real-time demo...")
    input()
    
    # Demo parameters
    exercise = 'squat'
    rep_count = 0
    start_time = time.time()
    frame_count = 0
    last_feedback_time = 0
    
    print("\n🚀 Starting real-time pose detection and analysis...")
    print("Watch how EdgeCoach analyzes your exercise form in real-time!")
    print("\nPress Ctrl+C to stop the demo")
    
    try:
        while True:
            clear_screen()
            show_header()
            
            # Simulate real-time processing
            frame_count += 1
            current_time = time.time()
            
            # Get pose data
            pose_data = simulate_pose_detection(exercise, frame_count)
            
            # Analyze form
            if exercise == 'squat':
                analysis = analyze_squat_form(pose_data, rep_count)
                rep_count = analysis['rep_count']
            elif exercise == 'plank':
                hold_time = current_time - start_time
                analysis = analyze_plank_form(pose_data, hold_time)
            else:
                analysis = {'feedback': None, 'quality': 'unknown', 'rep_count': 0}
            
            # Show pose overlay
            show_pose_overlay(pose_data, analysis, frame_count, 30.0, 15.0)
            
            # Show exercise analysis
            show_exercise_analysis(exercise, analysis, frame_count)
            
            # Show performance metrics
            show_performance_metrics(30.0, 15.0, frame_count)
            
            # Show controls
            show_controls()
            
            # Simulate voice feedback
            if analysis.get('feedback') and current_time - last_feedback_time > 2.0:
                print(f"🔊 VOICE: {analysis['feedback']}")
                last_feedback_time = current_time
            
            # Simulate real-time delay
            time.sleep(0.5)  # 2 FPS for demo
            
            # Auto-switch exercises for demo
            if frame_count % 50 == 0:
                if exercise == 'squat':
                    exercise = 'plank'
                    start_time = current_time
                    print("🔄 Auto-switching to Plank exercise...")
                else:
                    exercise = 'squat'
                    rep_count = 0
                    print("🔄 Auto-switching to Squat exercise...")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
        print("Thanks for watching EdgeCoach in action!")
        print("\nThis demo showed:")
        print("• Real-time pose detection and tracking")
        print("• Exercise form analysis with feedback")
        print("• Voice coaching system")
        print("• Performance monitoring")
        print("• Complete edge AI functionality")

if __name__ == "__main__":
    main()
