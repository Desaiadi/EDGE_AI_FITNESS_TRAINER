# Edge AI Fitness Trainer - Snapdragon Setup Guide

## System Requirements

- Snapdragon X Elite powered device
- Windows 11 (recommended)
- Python 3.8+ (3.11 recommended)
- Git
- Camera/webcam access
- 8GB RAM (minimum)
- 2GB free storage

## Setup Instructions

### 1. Install Required Software

```bash
# Install Python (if not already installed)
# Download from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH" during installation

# Install Git (if not already installed)
# Download from https://git-scm.com/downloads
```

### 2. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Desaiadi/EDGE_AI_FITNESS_TRAINER.git
cd EDGE_AI_FITNESS_TRAINER

# Switch to the Adi-update branch
git checkout Adi-update
```

### 3. Set Up Qualcomm Neural Network SDK

1. Download the Qualcomm Neural Network SDK:
   - Visit [Qualcomm Developer Network](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
   - Download the SDK for Windows
   - Run the installer and follow the setup wizard

2. Set Environment Variables:
   ```bash
   # Add these to your system environment variables
   QNN_SDK_ROOT=C:\path\to\qnn\sdk
   PATH=%PATH%;%QNN_SDK_ROOT%\bin
   ```

### 4. Install Python Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install QNN backend for ONNX Runtime
pip install onnxruntime-qnn
```

### 5. Verify NPU Setup

Run the test script to verify NPU acceleration:
```bash
python test_local_llm.py
```

You should see output indicating that the QNNExecutionProvider is available and active.

### 6. Common Issues and Solutions

1. **QNN Provider Not Found**:
   - Ensure QNN SDK is properly installed
   - Verify environment variables are set
   - Check if onnxruntime-qnn is installed

2. **Model Loading Errors**:
   - The first run will download and quantize models
   - Ensure you have write permissions in the models directory
   - Check disk space availability

3. **Performance Issues**:
   - Enable NPU performance mode in Windows settings
   - Close unnecessary background applications
   - Monitor system temperature

### 7. Optimizing for Snapdragon

The application automatically:
- Detects NPU availability
- Optimizes models for NPU execution
- Uses VTCM (Vector Tiling Cache Memory)
- Implements efficient memory patterns

To maximize performance:
1. Keep the device plugged in
2. Use "Performance" power mode
3. Ensure good ventilation
4. Close other GPU/NPU intensive applications

### 8. Running the Application

```bash
# Activate virtual environment (if using)
.\venv\Scripts\activate  # On Windows

# Run the main application
python edge_ai_trainer.py
```

### 9. Monitoring NPU Usage

1. Open Windows Task Manager
2. Go to Performance tab
3. Look for NPU utilization
4. Monitor temperature and power usage

### 10. Development Tips

1. Use Visual Studio Code with Python extension
2. Install Pylint for code quality
3. Use the debugger to monitor NPU operations
4. Check logs in the models directory

### 11. Updating the Application

```bash
# Get latest changes
git pull origin Adi-update

# Update dependencies if needed
pip install -r requirements.txt --upgrade
```

## Troubleshooting

### NPU Not Detected
```python
# Check available providers
import onnxruntime as ort
print(ort.get_available_providers())
```

### Performance Issues
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Model Quantization
```python
# Force model re-quantization
import os
os.remove('models/fitness_llm_quantized.onnx')
```

## Support

For issues and questions:
1. Check the [GitHub Issues](https://github.com/Desaiadi/EDGE_AI_FITNESS_TRAINER/issues)
2. Review Qualcomm Developer documentation
3. Contact the development team

## Performance Metrics

Expected performance on Snapdragon X Elite:
- Inference time: <50ms
- NPU utilization: 60-80%
- Memory usage: ~2GB
- Temperature: <70°C under load

## Security Notes

1. All processing happens locally
2. No data leaves the device
3. Models are encrypted at rest
4. Regular security updates recommended
