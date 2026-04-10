@echo off
chcp 65001 >nul
title Setup - Qwen3.5-2B

echo ============================================
echo   SETUP QWEN3.5-2B (Chi can chay 1 lan)
echo ============================================
echo.

echo [1/3] Cai Python packages...
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo Phat hien NVIDIA GPU. Thu cai ban CUDA 12.1...
    pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
    if %errorlevel% neq 0 (
        echo Cai ban GPU that bai. Chuyen sang ban CPU...
        pip install -q llama-cpp-python
    )
) else (
    echo Khong tim thay NVIDIA GPU. Cai ban CPU...
    pip install -q llama-cpp-python
)

if %errorlevel% neq 0 (
    echo Cai llama-cpp-python that bai.
    pause
    exit /b 1
)

pip install -q fastapi "uvicorn[standard]" huggingface_hub python-dotenv docling
if %errorlevel% neq 0 (
    echo Cai Python dependencies that bai.
    pause
    exit /b 1
)
echo Xong!

echo.
echo [2/3] Tai model Qwen3.5-2B GGUF...
if not exist "..\models" mkdir "..\models"

python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='unsloth/Qwen3.5-2B-GGUF', filename='Qwen3.5-2B-Q4_K_M.gguf', local_dir='../models')"
if %errorlevel% neq 0 (
    echo Tai model that bai. Kiem tra ket noi mang.
    pause
    exit /b 1
)
echo Model da tai xong!

echo.
echo [3/3] Xong!
echo ============================================
echo   Setup hoan tat. Chay start.bat de bat dau.
echo ============================================
pause
