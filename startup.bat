@echo off
setlocal

rem Start from the repository even when this file is launched by double-clicking it.
cd /d "%~dp0"

set "NINFER_DOCKER_IMAGE=ninfer-rtx50-dev:cuda13.1"
set "NINFER_MODEL_DIR=D:\AiModels\NInfer"
set "NINFER_MODEL_FILE=%NINFER_MODEL_DIR%\qwen3_8_27b.ninfer"
set "NINFER_HOST_PORT=8085"
set "NINFER_API_KEY=local-secret"

where docker >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or is not available on PATH.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running or its engine is unavailable.
    pause
    exit /b 1
)

if not exist "%NINFER_MODEL_FILE%" (
    echo ERROR: Model artifact not found:
    echo        %NINFER_MODEL_FILE%
    pause
    exit /b 1
)

if not exist "build-rtx50-linux\apps\ninfer-serve" (
    echo ERROR: NInfer server binary is missing:
    echo        %CD%\build-rtx50-linux\apps\ninfer-serve
    echo Build the codex/perf-rtx50-dual-gpu branch before running this file.
    pause
    exit /b 1
)

docker image inspect "%NINFER_DOCKER_IMAGE%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker image "%NINFER_DOCKER_IMAGE%" is missing.
    echo Build it with:
    echo docker build -f experiments/rtx50/Dockerfile -t %NINFER_DOCKER_IMAGE% .
    pause
    exit /b 1
)

if /i "%~1"=="--check" (
    echo Startup prerequisites are available.
    echo Endpoint: http://127.0.0.1:%NINFER_HOST_PORT%/v1
    exit /b 0
)

echo Starting Qwen3.8-27B on RTX 5070 Ti device 0 and RTX 5060 Ti device 1...
echo OpenAI base URL: http://127.0.0.1:%NINFER_HOST_PORT%/v1
echo Model ID:        qwen3.8-27b
echo API key:         %NINFER_API_KEY%
echo Press Ctrl+C to stop the server.
echo.

docker run --rm --gpus all ^
  --publish 127.0.0.1:%NINFER_HOST_PORT%:8080 ^
  --volume "%CD%:/workspace" ^
  --volume "%NINFER_MODEL_DIR%:/models:ro" ^
  --workdir /workspace ^
  "%NINFER_DOCKER_IMAGE%" ^
  ./build-rtx50-linux/apps/ninfer-serve /models/qwen3_8_27b.ninfer ^
  --host 0.0.0.0 --port 8080 ^
  --api-key "%NINFER_API_KEY%" --model-id qwen3.8-27b ^
  --device 0 --endpoint-device 1 ^
  --max-context 2048 --kv-capacity 2048 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3 --lm-head-draft

set "NINFER_EXIT_CODE=%ERRORLEVEL%"
if not "%NINFER_EXIT_CODE%"=="0" (
    echo.
    echo NInfer exited with code %NINFER_EXIT_CODE%.
    pause
)
exit /b %NINFER_EXIT_CODE%
