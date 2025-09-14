#!/usr/bin/env python3
"""
Test different camera backends
"""

import cv2
import time

def test_camera_backend(backend):
    """Test camera with specific backend"""
    print(f"🎥 Testing camera with backend: {backend}")
    
    # Open camera with specific backend
    cap = cv2.VideoCapture(0, backend)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera with backend {backend}")
        return False
    
    print(f"✅ Camera opened with backend {backend}")
    
    # Try to capture a frame
    ret, frame = cap.read()
    
    if ret and frame is not None:
        print(f"✅ Successfully captured frame with backend {backend}")
        print(f"Frame shape: {frame.shape}")
        
        # Save test image
        filename = f'camera_test_{backend}.jpg'
        cv2.imwrite(filename, frame)
        print(f"✅ Test image saved as '{filename}'")
        
        cap.release()
        return True
    else:
        print(f"❌ Failed to capture frame with backend {backend}")
        cap.release()
        return False

def main():
    print("🎥 Camera Backend Test")
    print("=" * 30)
    
    # Test different backends
    backends = [
        cv2.CAP_DSHOW,      # DirectShow
        cv2.CAP_MSMF,       # Microsoft Media Foundation
        cv2.CAP_ANY,        # Any available
        0,                  # Default
    ]
    
    backend_names = {
        cv2.CAP_DSHOW: "DirectShow",
        cv2.CAP_MSMF: "Microsoft Media Foundation", 
        cv2.CAP_ANY: "Any Available",
        0: "Default"
    }
    
    working_backend = None
    
    for backend in backends:
        try:
            if test_camera_backend(backend):
                working_backend = backend
                print(f"🎉 Found working backend: {backend_names[backend]}")
                break
        except Exception as e:
            print(f"❌ Error with backend {backend}: {e}")
    
    if working_backend is not None:
        print(f"\n✅ Camera is working with backend: {backend_names[working_backend]}")
        print("This backend can be used for the web application!")
    else:
        print("\n❌ No working camera backend found")
        print("Camera may be in use by another application")

if __name__ == "__main__":
    main()
