#!/usr/bin/env python3
"""
Verify camera is working and save a test image
"""

import cv2
import time

def main():
    print("🎥 Camera Verification Test")
    print("=" * 30)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened successfully")
    print("Capturing test image...")
    
    # Capture a frame
    ret, frame = cap.read()
    
    if ret:
        print("✅ Successfully captured frame")
        print(f"Frame shape: {frame.shape}")
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Add text
        cv2.putText(frame, "EdgeCoach Camera Test", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "This is your actual camera feed!", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save test image
        cv2.imwrite('camera_test_image.jpg', frame)
        print("✅ Test image saved as 'camera_test_image.jpg'")
        
        # Show frame briefly
        cv2.imshow('Camera Test - Press any key to close', frame)
        cv2.waitKey(3000)  # Wait 3 seconds
        cv2.destroyAllWindows()
        
    else:
        print("❌ Failed to capture frame")
    
    cap.release()
    print("✅ Camera released")

if __name__ == "__main__":
    main()
