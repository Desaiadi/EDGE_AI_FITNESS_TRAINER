"""
Build script for Edge AI Trainer - Creates standalone executable
Optimized for Snapdragon X Elite and Windows Copilot+ PCs
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "opencv-python==4.8.1.78",
        "mediapipe==0.10.7",
        "numpy==1.24.3",
        "Pillow==10.0.0",
        "onnxruntime==1.16.1",
        "pyinstaller==5.13.2",
        "requests==2.31.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {dep}")
            print(result.stderr)
        else:
            print(f"✅ Installed {dep}")

def create_spec_file():
    """Create PyInstaller spec file optimized for Edge AI"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect MediaPipe data files
mediapipe_data = collect_data_files('mediapipe')
cv2_data = collect_data_files('cv2')

# Hidden imports for AI/ML libraries
hiddenimports = [
    'cv2',
    'mediapipe',
    'numpy',
    'sqlite3',
    'tkinter',
    'tkinter.ttk',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'onnxruntime',
    'threading',
    'json',
    'datetime',
    'pathlib',
    'math',
    'requests'
] + collect_submodules('mediapipe') + collect_submodules('cv2')

a = Analysis(
    ['edge_ai_trainer.py'],
    pathex=[],
    binaries=[],
    datas=mediapipe_data + cv2_data + [
        ('README.md', '.'),
        ('LICENSE', '.') if os.path.exists('LICENSE') else None
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'IPython'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EdgeAITrainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app, no console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

# Create Windows App Package (.msix) - Optional
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EdgeAITrainer'
)
'''
    
    with open('edge_ai_trainer.spec', 'w') as f:
        f.write(spec_content.strip())
    
    print("✅ Created PyInstaller spec file")

def build_executable():
    """Build the standalone executable"""
    print("🔨 Building executable...")
    
    # Clean previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
    
    # Run PyInstaller
    result = subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "edge_ai_trainer.spec"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Successfully built executable!")
        print("📁 Executable location: dist/EdgeAITrainer.exe")
    else:
        print("❌ Build failed!")
        print(result.stderr)
        return False
    
    return True

def create_installer():
    """Create a simple installer script"""
    installer_script = '''
@echo off
echo Installing Edge AI Trainer...
echo.

REM Create application directory
if not exist "%PROGRAMFILES%\\EdgeAITrainer" mkdir "%PROGRAMFILES%\\EdgeAITrainer"

REM Copy files
xcopy "EdgeAITrainer.exe" "%PROGRAMFILES%\\EdgeAITrainer\\" /Y
xcopy "README.md" "%PROGRAMFILES%\\EdgeAITrainer\\" /Y

REM Create desktop shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%USERPROFILE%\\Desktop\\Edge AI Trainer.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%PROGRAMFILES%\\EdgeAITrainer\\EdgeAITrainer.exe" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo Edge AI Trainer installed successfully!
echo Desktop shortcut created.
pause
'''
    
    with open('dist/install.bat', 'w') as f:
        f.write(installer_script.strip())
    
    print("Created installer script: dist/install.bat")

def optimize_for_snapdragon():
    """Apply Snapdragon X Elite specific optimizations"""
    print("Applying Snapdragon X Elite optimizations...")
    
    # Create optimization config file
    config = {
        "npu_acceleration": True,
        "enable_quantization": True,
        "target_platform": "snapdragon_x_elite",
        "optimization_level": "high_performance",
        "memory_optimization": True,
        "threading": {
            "max_threads": 8,  # Typical for Snapdragon X Elite
            "cpu_affinity": "performance_cores"
        },
        "ai_acceleration": {
            "use_npu": True,
            "fallback_to_cpu": True,
            "model_format": "onnx",
            "precision": "int8"  # For better NPU performance
        }
    }
    
    import json
    with open('dist/snapdragon_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created Snapdragon optimization config")

def create_readme():
    """Create comprehensive README"""
    readme_content = '''# Edge AI Trainer - Snapdragon X Elite Edition

A comprehensive AI-powered fitness trainer that runs entirely on-device using Snapdragon X Elite NPU acceleration.

## Features

- **AI Fitness Planner**: Personalized workout and nutrition plans using local AI models
- **Computer Vision Trainer**: Real-time exercise form correction and rep counting
- **Edge Computing**: 100% offline operation with NPU acceleration
- **Exercise Tracking**: Support for push-ups, squats, planks with form analysis
- **Progress Monitoring**: Complete workout history and performance analytics

## System Requirements

- Windows 11 (Copilot+ PC)
- Snapdragon X Elite processor
- Camera/webcam
- 8GB RAM minimum
- 2GB storage space

## Installation

1. Download the latest release
2. Extract the ZIP file
3. Run `install.bat` as administrator (optional)
4. Or simply run `EdgeAITrainer.exe` directly

## Quick Start

1. **Create Profile**: Set up your fitness profile with age, weight, goals
2. **Generate Plans**: Use AI to create personalized workout and nutrition plans  
3. **Live Workout**: Start camera and select exercise for real-time tracking
4. **Track Progress**: Monitor your improvement over time

## Supported Exercises

- **Push-ups**: Elbow angle analysis and rep counting
- **Squats**: Knee angle and form checking
- **Planks**: Body alignment and duration tracking

## Edge AI Features

- Local LLM for fitness advice (no internet required)
- NPU-accelerated computer vision
- Privacy-first: all data stays on device
- Optimized for Snapdragon X Elite performance

## Hackathon Compliance

Built for the Qualcomm x NYU Edge AI Developer Hackathon:
- ✅ Edge-first architecture
- ✅ NPU utilization
- ✅ Complete offline operation
- ✅ Windows executable (.EXE)
- ✅ Open source code
- ✅ Local processing and privacy

## Technical Architecture

- **Frontend**: Tkinter GUI with modern styling
- **Computer Vision**: MediaPipe pose estimation
- **AI Processing**: ONNX Runtime with NPU acceleration
- **Database**: SQLite for local data storage
- **Packaging**: PyInstaller for standalone executable

## Development

### Prerequisites
```bash
pip install -r requirements.txt
```

### Building from Source
```bash
python build.py
```

### Running in Development
```bash
python edge_ai_trainer.py
```

## License

Open source under MIT License - see LICENSE file for details.

## Support

For hackathon support, contact the development team or refer to Qualcomm Developer Resources.
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content.strip())
    
    print("Created README.md")

def create_license():
    """Create MIT license file"""
    license_content = '''MIT License

Copyright (c) 2025 Edge AI Trainer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    
    with open('LICENSE', 'w') as f:
        f.write(license_content.strip())
    
    print("Created LICENSE file")

def main():
    """Main build process"""
    print("🚀 Building Edge AI Trainer for Snapdragon X Elite")
    print("=" * 50)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Create supporting files
    create_readme()
    create_license()
    
    # Step 3: Create PyInstaller spec
    create_spec_file()
    
    # Step 4: Build executable
    if build_executable():
        # Step 5: Create installer
        create_installer()
        
        # Step 6: Apply Snapdragon optimizations
        optimize_for_snapdragon()
        
        print("\n✅ Build completed successfully!")
        print("📦 Package contents:")
        print("   - EdgeAITrainer.exe (Main executable)")
        print("   - install.bat (Installer script)")
        print("   - snapdragon_config.json (NPU optimization)")
        print("   - README.md (Documentation)")
        print("\n🎯 Ready for Edge AI Developer Hackathon submission!")
        
        # Calculate approximate package size
        try:
            import os
            exe_size = os.path.getsize('dist/EdgeAITrainer.exe') / (1024*1024)
            print(f"📏 Executable size: {exe_size:.1f} MB")
        except:
            pass
            
    else:
        print("\n❌ Build failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()