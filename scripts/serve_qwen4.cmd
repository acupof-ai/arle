@echo off
setlocal
rem One-click ARLE chat server: Qwen3.8-Flash-Next (NVFP4) on the Vulkan lane.
rem Starts `arle serve` with the built-in chat UI at / and lets `--open`
rem launch the browser once the model is loaded and the port is bound.
rem Stop the server with Ctrl+C in this window.

set "ROOT=%~dp0.."
set "ARLE=%ROOT%\target\release\arle.exe"
set "MODEL=C:\Users\Asus\models\qwen3.8-flash-next-nvfp4"

if not exist "%ARLE%" (
  echo [serve_qwen4] release binary not found:
  echo [serve_qwen4]   %ARLE%
  echo [serve_qwen4] build it first, from the repo root:
  echo [serve_qwen4]   cargo build --release -p arle --features vulkan
  exit /b 1
)

if not exist "%MODEL%" (
  echo [serve_qwen4] model directory not found: %MODEL%
  exit /b 1
)

rem First free port from the candidate list; concurrent benchmark agents on
rem this box may already hold earlier ones.
set "PORT="
for %%P in (8080 8081 8082 8090) do (
  if not defined PORT (
    netstat -an | findstr /C:"LISTENING" | findstr /C:":%%P " >nul
    if errorlevel 1 set "PORT=%%P"
  )
)
if not defined PORT (
  echo [serve_qwen4] ports 8080 8081 8082 8090 are all busy; run manually:
  echo [serve_qwen4]   "%ARLE%" serve --backend vulkan --model-path "%MODEL%" --port NNNN --open
  exit /b 1
)

echo [serve_qwen4] serving on http://127.0.0.1:%PORT%/
echo [serve_qwen4] the browser opens after the model finishes loading; the
echo [serve_qwen4] checkpoint is ~126 GiB, so give it a minute.
"%ARLE%" serve --backend vulkan --model-path "%MODEL%" --port %PORT% --open
