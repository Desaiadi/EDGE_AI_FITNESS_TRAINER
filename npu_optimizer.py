"""
NPU Optimization Module for Snapdragon X Elite
Provides hardware acceleration for AI workloads using Qualcomm NPU
"""

import os
import json
import logging
from pathlib import Path
import numpy as np

# Try importing ONNX Runtime with NPU provider
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try importing Qualcomm AI Hub tools (if available)
try:
    # These would be available with Qualcomm AI Hub SDK
    # import qai_hub_models
    # import qai_appbuilder
    QAI_HUB_AVAILABLE = False  # Set to True when SDK is available
except ImportError:
    QAI_HUB_AVAILABLE = False

class SnapdragonNPUOptimizer:
    """Handles NPU acceleration for Snapdragon X Elite"""
    
    def __init__(self):
        self.npu_available = self._check_npu_availability()
        self.session_options = self._setup_session_options()
        self.providers = self._get_available_providers()
        self.config = self._load_optimization_config()
        
        logging.info(f"NPU Available: {self.npu_available}")
        logging.info(f"Available providers: {self.providers}")
    
    def _check_npu_availability(self):
        """Check if Snapdragon NPU is available"""
        if not ONNX_AVAILABLE:
            return False
        
        # Check for QNN (Qualcomm Neural Network) provider
        available_providers = ort.get_available_providers()
        return 'QNNExecutionProvider' in available_providers
    
    def _setup_session_options(self):
        """Configure ONNX Runtime session options for optimal performance"""
        if not ONNX_AVAILABLE:
            return None
            
        session_options = ort.SessionOptions()
        
        # Optimize for inference
        session_options.intra_op_num_threads = 8  # Snapdragon X Elite has 8+ cores
        session_options.inter_op_num_threads = 4
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Memory optimization
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        return session_options
    
    def _get_available_providers(self):
        """Get prioritized list of execution providers"""
        if not ONNX_AVAILABLE:
            return []
        
        available = ort.get_available_providers()
        
        # Priority order for Snapdragon X Elite
        preferred_order = [
            'QNNExecutionProvider',      # Qualcomm NPU (preferred)
            'DmlExecutionProvider',      # DirectML (GPU/NPU fallback)
            'CPUExecutionProvider'       # CPU (final fallback)
        ]
        
        # Return available providers in priority order
        return [p for p in preferred_order if p in available]
    
    def _load_optimization_config(self):
        """Load NPU optimization configuration"""
        config_path = Path('snapdragon_config.json')
        
        default_config = {
            "npu_acceleration": True,
            "enable_quantization": True,
            "target_platform": "snapdragon_x_elite",
            "optimization_level": "high_performance",
            "memory_optimization": True,
            "threading": {
                "max_threads": 8,
                "cpu_affinity": "performance_cores"
            },
            "ai_acceleration": {
                "use_npu": True,
                "fallback_to_cpu": True,
                "model_format": "onnx",
                "precision": "int8"
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                logging.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def create_optimized_session(self, model_path):
        """Create ONNX Runtime session with NPU optimization"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        
        try:
            # Configure provider options for QNN (Qualcomm NPU)
            provider_options = []
            
            if 'QNNExecutionProvider' in self.providers:
                qnn_options = {
                    'backend_path': 'QnnHtp.dll',  # High-performance backend
                    'profiling_level': 'basic',
                    'rpc_control_latency': '10',
                    'vtcm_mb': '8',  # Increase VTCM for better performance
                    'htp_performance_mode': 'burst'  # Max performance mode
                }
                provider_options.append(('QNNExecutionProvider', qnn_options))
            
            if 'DmlExecutionProvider' in self.providers:
                dml_options = {
                    'device_id': 0,
                    'enable_dynamic_graph_fusion': True
                }
                provider_options.append(('DmlExecutionProvider', dml_options))
            
            # Add CPU as fallback
            provider_options.append(('CPUExecutionProvider', {}))
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=provider_options
            )
            
            logging.info(f"Created session with providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            logging.error(f"Failed to create NPU session: {e}")
            # Fallback to CPU-only session
            return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    def optimize_model_for_npu(self, model_path, output_path=None):
        """Optimize ONNX model for Snapdragon NPU"""
        if not output_path:
            base_name = Path(model_path).stem
            output_path = f"{base_name}_optimized.onnx"
        
        try:
            # Model optimization techniques for NPU
            if QAI_HUB_AVAILABLE:
                # Use Qualcomm AI Hub optimization tools
                pass  # Would implement QAI Hub optimization here
            else:
                # Basic ONNX optimization
                import onnx
                from onnxruntime.tools import optimizer
                
                # Load and optimize model
                model = onnx.load(model_path)
                
                # Apply graph optimizations
                optimized_model = optimizer.optimize_graph(
                    model,
                    optimization_level=optimizer.OptimizationLevel.ORT_ENABLE_ALL,
                    disabled_optimizers=[]
                )
                
                # Save optimized model
                onnx.save(optimized_model, output_path)
                logging.info(f"Optimized model saved to: {output_path}")
                
                return output_path
                
        except Exception as e:
            logging.error(f"Model optimization failed: {e}")
            return model_path  # Return original if optimization fails
    
    def benchmark_performance(self, session, input_data, num_runs=100):
        """Benchmark model performance on NPU vs CPU"""
        import time
        
        results = {
            'npu_times': [],
            'avg_inference_time': 0,
            'throughput': 0,
            'provider_used': session.get_providers()[0]
        }
        
        # Warm up
        for _ in range(10):
            session.run(None, input_data)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            session.run(None, input_data)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        results['npu_times'] = times
        results['avg_inference_time'] = np.mean(times) * 1000  # Convert to ms
        results['throughput'] = 1.0 / np.mean(times)  # FPS
        results['std_dev'] = np.std(times) * 1000  # Standard deviation in ms
        
        logging.info(f"Benchmark Results:")
        logging.info(f"  Provider: {results['provider_used']}")
        logging.info(f"  Avg Inference Time: {results['avg_inference_time']:.2f}ms")
        logging.info(f"  Throughput: {results['throughput']:.1f} FPS")
        logging.info(f"  Std Dev: {results['std_dev']:.2f}ms")
        
        return results

class EdgeAIModelManager:
    """Manages AI models optimized for edge deployment"""
    
    def __init__(self):
        self.npu_optimizer = SnapdragonNPUOptimizer()
        self.models = {}
        self.model_cache_dir = Path("models")
        self.model_cache_dir.mkdir(exist_ok=True)
    
    def load_pose_estimation_model(self):
        """Load pose estimation model optimized for NPU"""
        # For this hackathon demo, we use MediaPipe's built-in models
        # In production, we would load custom ONNX models here
        
        try:
            # MediaPipe already handles hardware acceleration internally
            import mediapipe as mp
            
            pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Balance between accuracy and performance
                enable_segmentation=False,  # Disable for better performance
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.models['pose_estimation'] = pose_model
            logging.info("Loaded MediaPipe Pose model with hardware acceleration")
            
            return pose_model
            
        except Exception as e:
            logging.error(f"Failed to load pose estimation model: {e}")
            return None
    
    def load_custom_fitness_model(self, model_path):
        """Load custom fitness-specific model for NPU"""
        if not Path(model_path).exists():
            logging.warning(f"Model not found: {model_path}")
            return None
        
        try:
            # Optimize model for NPU
            optimized_path = self.npu_optimizer.optimize_model_for_npu(model_path)
            
            # Create NPU-accelerated session
            session = self.npu_optimizer.create_optimized_session(optimized_path)
            
            self.models['fitness_classifier'] = session
            logging.info(f"Loaded custom fitness model: {model_path}")
            
            return session
            
        except Exception as e:
            logging.error(f"Failed to load custom model: {e}")
            return None
    
    def download_optimized_models(self):
        """Download pre-optimized models from Qualcomm AI Hub"""
        # This would integrate with Qualcomm AI Hub model zoo
        # For hackathon, we'll create placeholder functionality
        
        models_to_download = [
            "fitness_pose_classifier_int8.onnx",
            "exercise_form_analyzer.onnx",
            "nutrition_recommender_quantized.onnx"
        ]
        
        for model_name in models_to_download:
            model_path = self.model_cache_dir / model_name
            
            if not model_path.exists():
                logging.info(f"Would download {model_name} from AI Hub")
                # In production: download from Qualcomm AI Hub
                # For demo: create empty file as placeholder
                model_path.touch()
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            "npu_available": self.npu_optimizer.npu_available,
            "providers": self.npu_optimizer.providers,
            "loaded_models": list(self.models.keys()),
            "optimization_config": self.npu_optimizer.config
        }
        return info

# Utility functions for NPU acceleration
def setup_npu_logging():
    """Setup logging for NPU operations"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - NPU - %(levelname)s - %(message)s'
    )

def check_snapdragon_hardware():
    """Check if running on Snapdragon X Elite hardware"""
    try:
        import platform
        import subprocess
        
        # Check CPU info
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"], 
                capture_output=True, text=True
            )
            cpu_info = result.stdout.lower()
            
            if "snapdragon" in cpu_info or "qualcomm" in cpu_info:
                return True
        
        # Additional checks could be added here
        return False
        
    except Exception:
        return False

def get_hardware_capabilities():
    """Get detailed hardware capability information"""
    capabilities = {
        "snapdragon_x_elite": check_snapdragon_hardware(),
        "npu_available": False,
        "gpu_available": False,
        "memory_gb": 0,
        "cpu_cores": 0
    }
    
    try:
        import psutil
        capabilities["memory_gb"] = psutil.virtual_memory().total / (1024**3)
        capabilities["cpu_cores"] = psutil.cpu_count()
        
        if ONNX_AVAILABLE:
            providers = ort.get_available_providers()
            capabilities["npu_available"] = 'QNNExecutionProvider' in providers
            capabilities["gpu_available"] = any(p in providers for p in 
                ['DmlExecutionProvider', 'CUDAExecutionProvider'])
        
    except ImportError:
        pass
    
    return capabilities

# Initialize global NPU optimizer instance
npu_optimizer = SnapdragonNPUOptimizer()
model_manager = EdgeAIModelManager()

# Export main classes and functions
__all__ = [
    'SnapdragonNPUOptimizer',
    'EdgeAIModelManager', 
    'setup_npu_logging',
    'check_snapdragon_hardware',
    'get_hardware_capabilities',
    'npu_optimizer',
    'model_manager'
]