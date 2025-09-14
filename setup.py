"""
Setup script for EdgeCoach
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"  stdout: {e.stdout}")
        logger.error(f"  stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    packages = [
        "onnxruntime-directml==1.16.3",
        "opencv-python==4.8.1.78",
        "pyttsx3==2.90",
        "numpy==1.24.3",
        "Pillow==10.0.1"
    ]
    
    for package in packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            return False
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly"""
    logger.info("Verifying installation...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"✓ ONNX Runtime installed - Providers: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            logger.info("✓ DirectML provider available")
        else:
            logger.warning("⚠ DirectML provider not available - will use CPU")
        
    except ImportError as e:
        logger.error(f"✗ ONNX Runtime verification failed: {e}")
        return False
    
    try:
        import cv2
        logger.info(f"✓ OpenCV installed - Version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"✗ OpenCV verification failed: {e}")
        return False
    
    try:
        import pyttsx3
        logger.info("✓ pyttsx3 installed")
    except ImportError as e:
        logger.error(f"✗ pyttsx3 verification failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy installed - Version: {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy verification failed: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "docs",
        "dist",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "edgecoach.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """Main setup function"""
    setup_logging()
    
    logger.info("Setting up EdgeCoach environment...")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Verify installation
    if not verify_installation():
        logger.error("Installation verification failed")
        return False
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Run: python test_app.py (to test all components)")
    logger.info("2. Run: python main.py (to start EdgeCoach)")
    logger.info("3. Run: python build.py (to create executable)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
