# EdgeCoach - AI-Powered Fitness Form Coach

## Team Information
- **Team Name**: EdgeCoach
- **Team Members**: [Your Names and Emails]
- **Event**: NYU Edge AI Developer Hackathon
- **Submission Date**: September 14, 2024

## Project Description

EdgeCoach is a real-time AI-powered fitness form coach that runs entirely on-device using edge AI capabilities. The application provides instant feedback on exercise form for squats and planks, helping users maintain proper technique and prevent injuries.

### Key Features
- **Real-time Pose Estimation**: Uses ONNX Runtime with DirectML for NPU/GPU acceleration
- **Exercise Form Analysis**: Detects and corrects form issues for squats and planks
- **Voice Coaching**: Provides real-time audio feedback and instructions
- **Privacy-First**: All processing happens locally - no data leaves your device
- **Low Latency**: <100ms response time for real-time coaching
- **Offline Operation**: Works completely without internet connection

## Technical Implementation

### Edge AI Stack
- **ONNX Runtime**: DirectML execution provider for NPU/GPU acceleration
- **Pose Model**: MoveNet/BlazePose for lightweight pose estimation
- **Computer Vision**: OpenCV for real-time video processing
- **Text-to-Speech**: Windows SAPI for voice feedback
- **Performance**: Optimized for Snapdragon X Elite Copilot+ PC

### Performance Metrics
- **Latency**: <100ms end-to-end processing
- **FPS**: 30+ FPS on Snapdragon X Elite
- **NPU Utilization**: DirectML execution provider active
- **Memory**: <2GB RAM usage
- **Privacy**: 100% local processing, no network calls

## Installation & Run

### Prerequisites
- Windows 10/11 with DirectML support
- Camera access permissions
- Snapdragon X Elite Copilot+ PC (recommended)

### Quick Start
1. Download the latest release from GitHub
2. Extract the EdgeCoach folder
3. Run `EdgeCoach.exe`
4. Allow camera permissions when prompted
5. Select your exercise (Squat or Plank)
6. Follow the on-screen instructions

### Troubleshooting
- **Camera not working**: Check Windows camera permissions in Settings > Privacy > Camera
- **Low performance**: Ensure DirectML drivers are installed and NPU is available
- **Audio issues**: Verify Windows audio output is working
- **App won't start**: Run as administrator if needed

## Exercise Support

### Squat Analysis
- **Stance Detection**: Monitors foot width and positioning
- **Depth Analysis**: Tracks hip-to-knee angle for proper depth
- **Knee Tracking**: Ensures knees stay over toes
- **Back Alignment**: Monitors spine neutrality
- **Rep Counting**: Automatically counts completed reps

### Plank Analysis
- **Body Alignment**: Tracks head-shoulder-hip-ankle line
- **Hip Position**: Detects sagging or piking
- **Hold Timer**: Counts down plank duration
- **Form Corrections**: Real-time feedback for alignment issues

## Privacy & Security

EdgeCoach is designed with privacy as a core principle:
- **Local Processing**: All video data stays on your device
- **No Network Calls**: Zero data transmission to external servers
- **No Data Storage**: Video frames are processed in real-time and discarded
- **Offline Operation**: Works completely without internet connection

## Development Setup

### Requirements
```bash
pip install onnxruntime-directml opencv-python pyttsx3 numpy
```

### Running from Source
```bash
git clone https://github.com/yourusername/edgecoach.git
cd edgecoach
pip install -r requirements.txt
python main.py
```

## Demo Script (5 Minutes)

### 1. Problem & Users (60 seconds)
- Home fitness and rehabilitation need real-time form feedback
- Privacy concerns with cloud-based solutions
- EdgeCoach provides local, private, instant coaching

### 2. Live Demo (180 seconds)
- **Squat Exercise**: Show real-time pose overlay, form analysis, voice feedback
- **Plank Exercise**: Demonstrate alignment detection and hold timer
- **Performance**: Display FPS counter and latency metrics

### 3. Technical Highlights (60 seconds)
- NPU/GPU utilization via DirectML
- <100ms latency processing
- Complete offline operation
- Privacy-first architecture

### 4. Impact & Future (60 seconds)
- Potential for home fitness, physical therapy, sports training
- Scalable to additional exercises
- Path to commercial deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- [ONNX Runtime DirectML Documentation](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [MoveNet Pose Estimation](https://www.tensorflow.org/hub/tutorials/movenet)
- [Windows App Packaging Guide](https://docs.microsoft.com/en-us/windows/msix/)

## Contact

For questions or support, please contact the team at [your-email@example.com]
