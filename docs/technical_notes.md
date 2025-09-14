# EdgeCoach Technical Implementation Notes

## Latency Analysis

### Target Performance
- **End-to-end latency**: <100ms
- **Frame processing**: 30+ FPS
- **Voice feedback delay**: <50ms

### Optimization Strategies
1. **Frame Resizing**: Input frames resized to 256x320px for pose estimation
2. **Buffer Reuse**: Pre-allocated buffers to avoid memory allocation overhead
3. **Vectorized Operations**: NumPy operations for joint angle calculations
4. **DirectML Provider**: NPU/GPU acceleration via ONNX Runtime DirectML

### Performance Monitoring
```python
# Latency tracking implementation
start_time = time.perf_counter()
# ... processing ...
latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
```

## NPU/GPU Usage

### DirectML Configuration
- **Execution Provider**: DirectML (preferred over CPU)
- **Model Format**: ONNX optimized for DirectML
- **Memory Management**: Efficient tensor operations

### Hardware Utilization
- **Snapdragon X Elite**: NPU acceleration for pose inference
- **Memory**: <2GB RAM usage
- **Power**: Optimized for mobile/edge deployment

## Privacy Implementation

### Local Processing
- **Video Stream**: Camera input processed in real-time, no storage
- **No Network Calls**: Zero external API calls
- **Data Retention**: No persistent storage of user data
- **Offline Operation**: Complete functionality without internet

### Security Measures
- **Input Validation**: Camera permissions only
- **Sandboxed Execution**: No file system access beyond app directory
- **Memory Safety**: Automatic cleanup of video frames

## Demo Script

### 5-Minute Presentation Structure

#### 1. Problem Statement (60 seconds)
- Home fitness lacks real-time form feedback
- Privacy concerns with cloud-based solutions
- Need for instant, local coaching

#### 2. Live Demonstration (180 seconds)
- **Squat Exercise**:
  - Show pose overlay with keypoints
  - Demonstrate form analysis (depth, knee tracking, back alignment)
  - Voice feedback examples
- **Plank Exercise**:
  - Alignment detection
  - Hold timer and form corrections
  - Real-time feedback

#### 3. Technical Highlights (60 seconds)
- Performance metrics display
- NPU/GPU utilization
- Privacy demonstration (no network activity)
- Offline operation

#### 4. Impact & Future (60 seconds)
- Commercial applications
- Scalability to more exercises
- Integration potential

### Key Demo Points
- **Performance**: Show FPS counter and latency display
- **Accuracy**: Demonstrate form detection accuracy
- **Responsiveness**: Real-time voice feedback
- **Privacy**: Show no network activity during operation
