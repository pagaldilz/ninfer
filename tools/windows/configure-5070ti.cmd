@echo off
setlocal

set "NINFER_ROOT=%~dp0..\.."
for %%I in ("%NINFER_ROOT%") do set "NINFER_ROOT=%%~fI"

if not defined NINFER_VS_ROOT set "NINFER_VS_ROOT=C:\BuildTools"
if not defined NINFER_MSVC_VERSION set "NINFER_MSVC_VERSION=14.42"
if not defined NINFER_BUILD_DIR set "NINFER_BUILD_DIR=%NINFER_ROOT%\build-win-5070ti"
if not defined NINFER_BUILD_BENCHMARKS set "NINFER_BUILD_BENCHMARKS=OFF"

call "%NINFER_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%NINFER_MSVC_VERSION% >nul
if errorlevel 1 exit /b %errorlevel%

set "CMAKE=%NINFER_VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%NINFER_VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "HOST_CL=%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"
set "NINJA_CMAKE=%NINJA:\=/%"
set "HOST_CL_CMAKE=%HOST_CL:\=/%"

"%CMAKE%" -S "%NINFER_ROOT%" -B "%NINFER_BUILD_DIR%" -G Ninja ^
  "-DCMAKE_MAKE_PROGRAM=%NINJA_CMAKE%" ^
  "-DCMAKE_C_COMPILER=%HOST_CL_CMAKE%" ^
  "-DCMAKE_CXX_COMPILER=%HOST_CL_CMAKE%" ^
  "-DCMAKE_CUDA_HOST_COMPILER=%HOST_CL_CMAKE%" ^
  "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DNINFER_BUILD_APPS=OFF ^
  -DBUILD_TESTING=OFF ^
  -DNINFER_BUILD_BENCHMARKS=%NINFER_BUILD_BENCHMARKS% ^
  -DNINFER_BUILD_TOOLS=OFF
if errorlevel 1 exit /b %errorlevel%

"%CMAKE%" --build "%NINFER_BUILD_DIR%" --parallel
