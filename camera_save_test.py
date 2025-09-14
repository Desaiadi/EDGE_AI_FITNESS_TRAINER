#!/usr/bin/env python3
"""
Save camera frames to verify camera is working
"""

import cv2
import time
import os

def main():
    print("🎥 Camera Save Test")
    print("=" * 30)
    print("This will save camera frames to verify the camera is working")
    
    # Create output directory
    os.makedirs('camera_frames', exist_ok=True)
    
    # Open camera with DirectShow backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    print("Saving camera frames...")
    print("Press Ctrl+C to stop")
    
    frame_count = 0
    
    try:
        while frame_count < 50:  # Save 50 frames
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Add text
                cv2.putText(frame, f"Frame {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "EdgeCoach Camera Test", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame
                filename = f'camera_frames/frame_{frame_count:03d}.jpg'
                cv2.imwrite(filename, frame)
                
                print(f"📹 Saved frame {frame_count}")
                
                # Show frame briefly
                cv2.imshow('Camera Test - Press any key to close', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            else:
                print("❌ Failed to capture frame")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")
        print(f"📁 Saved {frame_count} frames to 'camera_frames' folder")
        print("Check the 'camera_frames' folder to see your actual camera images!")

if __name__ == "__main__":
    main()
