"""
Build script for EdgeCoach
Creates a Windows executable package
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def install_pyinstaller():
    """Install PyInstaller if not available"""
    try:
        import PyInstaller
        logger.info("✓ PyInstaller already installed")
        return True
    except ImportError:
        logger.info("Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            logger.info("✓ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install PyInstaller: {e}")
            return False

def create_spec_file():
    """Create PyInstaller spec file for EdgeCoach"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('docs', 'docs'),
        ('README.md', '.'),
        ('LICENSE', '.'),
    ],
    hiddenimports=[
        'onnxruntime',
        'onnxruntime.directml',
        'cv2',
        'pyttsx3',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EdgeCoach',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
    
    with open('EdgeCoach.spec', 'w') as f:
        f.write(spec_content)
    
    logger.info("✓ PyInstaller spec file created")

def build_executable():
    """Build the executable using PyInstaller"""
    logger.info("Building EdgeCoach executable...")
    
    try:
        # Run PyInstaller
        cmd = [sys.executable, "-m", "PyInstaller", "--clean", "EdgeCoach.spec"]
        subprocess.check_call(cmd)
        
        logger.info("✓ Executable built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Build failed: {e}")
        return False

def create_installer():
    """Create installer package"""
    logger.info("Creating installer package...")
    
    # Create dist directory structure
    dist_dir = Path("dist/EdgeCoach")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy executable
    exe_src = Path("dist/EdgeCoach.exe")
    exe_dst = dist_dir / "EdgeCoach.exe"
    
    if exe_src.exists():
        shutil.copy2(exe_src, exe_dst)
        logger.info("✓ Executable copied to package")
    else:
        logger.error("✗ Executable not found")
        return False
    
    # Copy additional files
    files_to_copy = [
        ("README.md", "README.md"),
        ("LICENSE", "LICENSE"),
        ("docs", "docs"),
    ]
    
    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = dist_dir / dst
        
        if src_path.exists():
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
            logger.info(f"✓ Copied {src} to package")
        else:
            logger.warning(f"⚠ {src} not found, skipping")
    
    # Create batch file for easy execution
    batch_content = '''@echo off
echo Starting EdgeCoach...
EdgeCoach.exe
pause
'''
    
    with open(dist_dir / "run.bat", 'w') as f:
        f.write(batch_content)
    
    logger.info("✓ Installer package created")
    return True

def create_zip_package():
    """Create ZIP package for distribution"""
    logger.info("Creating ZIP package...")
    
    try:
        import zipfile
        
        zip_path = "EdgeCoach.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk("dist/EdgeCoach"):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, "dist")
                    zipf.write(file_path, arc_path)
        
        logger.info(f"✓ ZIP package created: {zip_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create ZIP package: {e}")
        return False

def main():
    """Main build function"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Building EdgeCoach for Windows...")
    logger.info("=" * 50)
    
    # Step 1: Install PyInstaller
    if not install_pyinstaller():
        return False
    
    # Step 2: Create spec file
    create_spec_file()
    
    # Step 3: Build executable
    if not build_executable():
        return False
    
    # Step 4: Create installer package
    if not create_installer():
        return False
    
    # Step 5: Create ZIP package
    if not create_zip_package():
        return False
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Build completed successfully!")
    logger.info("\nOutput files:")
    logger.info("  - dist/EdgeCoach/EdgeCoach.exe (executable)")
    logger.info("  - EdgeCoach.zip (distribution package)")
    logger.info("\nTo test the build:")
    logger.info("  cd dist/EdgeCoach")
    logger.info("  EdgeCoach.exe")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
