@echo off
setlocal

cd /d "%~dp0\.."

set "COMPOSE=docker compose --env-file .env -f docker-compose.runtime.yml"

echo.
echo [1/6] Starting Docker runtime...
%COMPOSE% up -d --build
if errorlevel 1 goto :fail

echo.
echo [2/6] Container status...
%COMPOSE% ps
if errorlevel 1 goto :fail

echo.
echo [3/6] API health...
for /l %%i in (1,1,30) do (
  curl.exe -fsS http://localhost:8000/api/v1/health >nul 2>nul
  if not errorlevel 1 goto :api_healthy
  ping 127.0.0.1 -n 3 >nul
)
echo API did not become healthy in time.
goto :fail

:api_healthy
curl.exe -fsS http://localhost:8000/api/v1/health
if errorlevel 1 goto :fail
echo.

echo.
echo [4/6] Online API torch/CUDA check...
docker exec aic-online-api-runtime python -c "import os, torch; device=os.getenv('CLIP_MODEL_DEVICE','cpu'); print('CLIP_MODEL_DEVICE=' + device); print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); assert device != 'cuda' or torch.cuda.is_available(), 'CLIP_MODEL_DEVICE=cuda but CUDA is not available inside online-api container'"
if errorlevel 1 goto :fail

echo.
echo [5/6] PostgreSQL to FAISS sync...
%COMPOSE% --profile tools run --rm --build data-tools python -m BackEnd.scripts.check_faiss_db_sync
if errorlevel 1 goto :fail

echo.
echo [6/6] Elasticsearch OCR smoke search...
%COMPOSE% --profile tools run --rm --build data-tools python scripts/search_es_direct.py tin --source ocr --top-k 1
if errorlevel 1 goto :fail

echo.
echo All runtime data checks passed.
exit /b 0

:fail
echo.
echo Runtime check failed. Check the error above.
exit /b 1
