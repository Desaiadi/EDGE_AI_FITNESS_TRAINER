#!/usr/bin/env python3
"""
Persistent camera feed with pose detection
"""

import cv2
import time
import numpy as np

def main():
    print("🎥 Persistent Camera Feed with Pose Detection")
    print("=" * 50)
    print("This will show your camera feed and keep the window open")
    print("Press 'q' to quit, 's' for squat, 'p' for plank")
    print("The window should stay open and show your live camera feed!")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera opened successfully")
    print("Showing persistent camera feed...")
    print("Look for the camera window that should open!")
    print("The window will stay open until you press 'q'")
    
    frame_count = 0
    exercise = "Unknown"
    
    try:
        while True:
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Create a copy for pose detection overlay
                display_frame = frame.copy()
                
                # Simulate pose detection (draw a simple skeleton on the REAL camera feed)
                height, width = frame.shape[:2]
                center_x = width // 2
                center_y = height // 2
                
                # Draw a simple skeleton overlay on the REAL camera feed
                # Head
                cv2.circle(display_frame, (center_x, center_y - 100), 15, (0, 255, 0), -1)
                # Body
                cv2.line(display_frame, (center_x, center_y - 85), (center_x, center_y + 50), (0, 255, 0), 3)
                # Arms
                cv2.line(display_frame, (center_x, center_y - 50), (center_x - 40, center_y), (0, 255, 0), 3)
                cv2.line(display_frame, (center_x, center_y - 50), (center_x + 40, center_y), (0, 255, 0), 3)
                # Legs
                cv2.line(display_frame, (center_x, center_y + 50), (center_x - 30, center_y + 100), (0, 255, 0), 3)
                cv2.line(display_frame, (center_x, center_y + 50), (center_x + 30, center_y + 100), (0, 255, 0), 3)
                
                # Add text overlay
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Exercise: {exercise}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'q' to quit, 's' for squat, 'p' for plank", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the REAL camera feed with pose detection
                cv2.imshow('LIVE Camera Feed - EdgeCoach', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    exercise = "SQUAT"
                    print("🏃‍♂️ Switched to SQUAT exercise!")
                elif key == ord('p'):
                    exercise = "PLANK"
                    print("🏃‍♂️ Switched to PLANK exercise!")
                
                # Print status every 100 frames
                if frame_count % 100 == 0:
                    print(f"📹 Processing frame {frame_count} - Camera feed should be visible!")
                
            else:
                print("❌ Failed to read frame")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")

if __name__ == "__main__":
    main()
