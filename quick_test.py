#!/usr/bin/env python3
"""
Quick test script for EdgeCoach
Tests basic functionality without requiring camera or model
"""

import sys
import logging
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports"""
    logger.info("Testing basic imports...")
    
    try:
        import cv2
        logger.info("✓ OpenCV imported")
    except ImportError:
        logger.error("✗ OpenCV not found - run: pip install opencv-python")
        return False
    
    try:
        import onnxruntime as ort
        logger.info("✓ ONNX Runtime imported")
        logger.info(f"  Available providers: {ort.get_available_providers()}")
    except ImportError:
        logger.error("✗ ONNX Runtime not found - run: pip install onnxruntime-directml")
        return False
    
    try:
        import pyttsx3
        logger.info("✓ pyttsx3 imported")
    except ImportError:
        logger.error("✗ pyttsx3 not found - run: pip install pyttsx3")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported")
    except ImportError:
        logger.error("✗ NumPy not found - run: pip install numpy")
        return False
    
    return True

def test_voice_system():
    """Test voice system"""
    logger.info("Testing voice system...")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        # Test basic voice functionality
        engine.say("EdgeCoach voice test")
        engine.runAndWait()
        
        logger.info("✓ Voice system working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Voice system failed: {e}")
        return False

def test_exercise_engine():
    """Test exercise engine with dummy data"""
    logger.info("Testing exercise engine...")
    
    try:
        from exercise_engine import ExerciseEngine
        
        engine = ExerciseEngine()
        engine.set_exercise('squat')
        
        # Create dummy keypoints (17 keypoints, 3 values each: x, y, confidence)
        dummy_keypoints = np.random.rand(17, 3) * 100
        
        # Test analysis
        analysis = engine.analyze_pose(dummy_keypoints)
        
        logger.info(f"✓ Exercise engine working - Analysis: {list(analysis.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Exercise engine failed: {e}")
        return False

def test_ui_overlay():
    """Test UI overlay with dummy data"""
    logger.info("Testing UI overlay...")
    
    try:
        from ui_overlay import UIOverlay
        
        overlay = UIOverlay()
        
        # Create dummy analysis
        dummy_analysis = {
            'exercise': 'squat',
            'state': 'setup',
            'rep_count': 5,
            'quality': 'good',
            'feedback': 'Keep your knees over your toes',
            'metrics': {
                'depth_angle': 95.5,
                'stance_width': 1.1,
                'knee_over_toe': 0.2,
                'back_angle': 88.0
            }
        }
        
        overlay.update_analysis(dummy_analysis)
        
        # Create dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test overlay drawing
        result = overlay.draw_overlay(dummy_frame)
        
        logger.info("✓ UI overlay working")
        return True
        
    except Exception as e:
        logger.error(f"✗ UI overlay failed: {e}")
        return False

def test_pose_estimator():
    """Test pose estimator initialization"""
    logger.info("Testing pose estimator...")
    
    try:
        from pose_estimator import PoseEstimator
        
        estimator = PoseEstimator()
        
        # Test basic functionality without model
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # This will fail without the actual model, but we can test the class
        logger.info("✓ Pose estimator class created")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pose estimator failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("EdgeCoach Quick Test")
    logger.info("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Voice System", test_voice_system),
        ("Exercise Engine", test_exercise_engine),
        ("UI Overlay", test_ui_overlay),
        ("Pose Estimator", test_pose_estimator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info("\n" + "=" * 40)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! EdgeCoach is ready.")
        logger.info("\nTo run EdgeCoach:")
        logger.info("  python main.py")
        logger.info("  or double-click run_edgecoach.bat")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed.")
        logger.error("\nTo fix issues:")
        logger.error("  python setup.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
