#!/usr/bin/env python3
"""
Simple video display that should work on Windows
"""

import cv2
import time

def main():
    print("🎥 Simple Video Display Test")
    print("=" * 30)
    print("This will test if we can display video on your system")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    print("Trying to display video...")
    
    frame_count = 0
    
    try:
        while frame_count < 100:  # Only run for 100 frames to test
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Add simple text
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press any key to close", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Video Test', frame)
                
                # Wait for key press
                key = cv2.waitKey(30) & 0xFF
                if key != 255:  # Any key pressed
                    break
                
                print(f"📹 Displaying frame {frame_count}")
                
            else:
                print("❌ Failed to read frame")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")

if __name__ == "__main__":
    main()
