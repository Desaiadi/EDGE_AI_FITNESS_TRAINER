#!/usr/bin/env python3
"""
Simple camera display without pose detection
"""

import cv2
import time

def main():
    print("🎥 Simple Camera Display")
    print("=" * 30)
    print("This will show your camera feed directly")
    print("Press 'q' to quit")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    print("Showing camera feed...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Your Camera Feed', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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
