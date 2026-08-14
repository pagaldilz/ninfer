@echo off
setlocal

if "%~1"=="" (
  echo Usage: %~nx0 ^<qwen3_6_35b_a3b.ninfer^>
  exit /b 2
)

set "NINFER_ROOT=%~dp0..\.."
for %%I in ("%NINFER_ROOT%") do set "NINFER_ROOT=%%~fI"
set "BENCH=%NINFER_ROOT%\build-win-5070ti-bench\bench\ninfer_bench.exe"
set "CORPUS=%NINFER_ROOT%\bench\fixtures\bench_corpus.ids"
set "OUTPUT=%NINFER_ROOT%\profiles\bench"

if not exist "%BENCH%" (
  echo Missing %BENCH%
  echo Build it with tools\windows\configure-5070ti.cmd and NINFER_BUILD_BENCHMARKS=ON.
  exit /b 2
)

"%BENCH%" --weights "%~f1" --corpus "%CORPUS%" -n 128 -r 2 --warmup 1 ^
  --max-ctx 32768 --prefill-chunk 128 --kv-dtype int8 --mtp-draft-tokens 1 ^
  --lm-head-draft --no-cuda-graph --text-only -o json ^
  --output-file "%OUTPUT%\win-5070ti-final-32k-decode.json"
if errorlevel 1 exit /b %errorlevel%

"%BENCH%" --weights "%~f1" --corpus "%CORPUS%" -pg 30615,128 -r 1 --warmup 0 ^
  --max-ctx 32768 --prefill-chunk 4096 --kv-dtype int8 --mtp-draft-tokens 2 ^
  --lm-head-draft --no-cuda-graph --text-only -o json ^
  --output-file "%OUTPUT%\win-5070ti-final-32k-prefill-decode.json"
exit /b %errorlevel%
