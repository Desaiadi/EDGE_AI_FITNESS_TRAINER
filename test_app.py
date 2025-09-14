"""
Test script for EdgeCoach
Verifies all components are working correctly
"""

import sys
import logging
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import onnxruntime as ort
        logger.info("✓ ONNX Runtime imported successfully")
        logger.info(f"  Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        logger.error(f"✗ Failed to import ONNX Runtime: {e}")
        return False
    
    try:
        import pyttsx3
        logger.info("✓ pyttsx3 imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import pyttsx3: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import NumPy: {e}")
        return False
    
    return True

def test_onnx_runtime():
    """Test ONNX Runtime configuration"""
    try:
        import onnxruntime as ort
        
        # Check available providers
        providers = ort.get_available_providers()
        logger.info(f"Available providers: {providers}")
        
        # Check if DirectML is available
        if 'DmlExecutionProvider' in providers:
            logger.info("✓ DirectML provider available")
        else:
            logger.warning("⚠ DirectML provider not available - will use CPU")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ONNX Runtime test failed: {e}")
        return False

def test_voice_system():
    """Test voice system"""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        voices = engine.get_property('voices')
        logger.info(f"✓ Voice system initialized with {len(voices)} voices")
        
        # Test speech
        engine.say("EdgeCoach voice system test")
        engine.runAndWait()
        
        engine.stop()
        logger.info("✓ Voice system test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Voice system test failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            logger.error("✗ Failed to read from camera")
            cap.release()
            return False
        
        logger.info(f"✓ Camera test successful - frame shape: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"✗ Camera test failed: {e}")
        return False

def test_pose_estimator():
    """Test pose estimator initialization"""
    try:
        from pose_estimator import PoseEstimator
        
        estimator = PoseEstimator()
        # Note: This will fail without the actual model file
        # but we can test the initialization logic
        logger.info("✓ Pose estimator class created")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pose estimator test failed: {e}")
        return False

def test_exercise_engine():
    """Test exercise engine"""
    try:
        from exercise_engine import ExerciseEngine
        
        engine = ExerciseEngine()
        engine.set_exercise('squat')
        
        # Test with dummy keypoints
        dummy_keypoints = np.random.rand(17, 3) * 100
        analysis = engine.analyze_pose(dummy_keypoints)
        
        logger.info("✓ Exercise engine test successful")
        logger.info(f"  Analysis keys: {list(analysis.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Exercise engine test failed: {e}")
        return False

def test_ui_overlay():
    """Test UI overlay"""
    try:
        from ui_overlay import UIOverlay
        
        overlay = UIOverlay()
        
        # Test with dummy analysis
        dummy_analysis = {
            'exercise': 'squat',
            'state': 'setup',
            'rep_count': 0,
            'quality': 'good',
            'feedback': 'Test feedback',
            'metrics': {'depth_angle': 90, 'stance_width': 1.0}
        }
        
        overlay.update_analysis(dummy_analysis)
        
        # Test overlay drawing
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = overlay.draw_overlay(dummy_frame)
        
        logger.info("✓ UI overlay test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ UI overlay test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Starting EdgeCoach component tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("ONNX Runtime Test", test_onnx_runtime),
        ("Voice System Test", test_voice_system),
        ("Camera Test", test_camera),
        ("Pose Estimator Test", test_pose_estimator),
        ("Exercise Engine Test", test_exercise_engine),
        ("UI Overlay Test", test_ui_overlay)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! EdgeCoach is ready to run.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    
    if success:
        logger.info("\nTo run EdgeCoach:")
        logger.info("  python main.py")
        sys.exit(0)
    else:
        logger.error("\nSome tests failed. Please fix the issues before running EdgeCoach.")
        sys.exit(1)

if __name__ == "__main__":
    main()
