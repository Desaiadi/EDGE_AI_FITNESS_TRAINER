#!/usr/bin/env python3
"""
Simple camera test to verify camera access
"""

import cv2
import sys

def test_camera():
    """Test camera access"""
    print("🎥 Testing camera access...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"✅ Camera {camera_index} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {camera_index} can read frames: {frame.shape}")
                
                # Show the camera feed
                print("Press 'q' to quit, any other key to try next camera")
                while True:
                    ret, frame = cap.read()
                    if ret:
                        # Resize frame for display
                        frame = cv2.resize(frame, (640, 480))
                        
                        # Add text overlay
                        cv2.putText(frame, f"Camera {camera_index} - Press 'q' to quit", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow(f'Camera {camera_index} Test', frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    else:
                        print(f"❌ Camera {camera_index} failed to read frame")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"❌ Camera {camera_index} cannot read frames")
                cap.release()
        else:
            print(f"❌ Camera {camera_index} cannot be opened")
    
    print("❌ No working camera found")
    return False

if __name__ == "__main__":
    print("🚀 Camera Test - EdgeCoach")
    print("=" * 40)
    
    if test_camera():
        print("✅ Camera test successful!")
    else:
        print("❌ Camera test failed!")
        print("Make sure your camera is not being used by another application")
    
    print("\nPress Enter to exit...")
    input()
