@echo off
echo ========================================
echo    EdgeCoach - AI Fitness Coach
echo ========================================
echo.
echo Starting EdgeCoach...
echo.
echo Controls:
echo   S - Switch to Squat exercise
echo   P - Switch to Plank exercise  
echo   R - Reset exercise
echo   Q - Quit application
echo.
echo ========================================
echo.

"%USERPROFILE%\AppData\Local\Programs\Python\Python312-arm64\python.exe" main.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Error: EdgeCoach failed to start
    echo ========================================
    echo.
    echo Troubleshooting:
    echo 1. Make sure Python is installed
    echo 2. Run: "%USERPROFILE%\AppData\Local\Programs\Python\Python312-arm64\python.exe" setup.py
    echo 3. Run: "%USERPROFILE%\AppData\Local\Programs\Python\Python312-arm64\python.exe" test_app.py
    echo.
    pause
)
