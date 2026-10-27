@echo off
rem Usage: tasks <command>     (run from the project root)
set PY=venv\Scripts\python

if "%~1"=="" goto help
if /i "%~1"=="train" goto train
if /i "%~1"=="test" goto test
if /i "%~1"=="serve" goto serve
if /i "%~1"=="mlflow-ui" goto mlflow_ui
if /i "%~1"=="docker-build" goto docker_build
if /i "%~1"=="docker-run" goto docker_run
if /i "%~1"=="help" goto help
echo Unknown command: %~1
goto help

:train
%PY% src/train.py
goto end

:test
%PY% -m pytest -v
goto end

:serve
%PY% -m uvicorn src.app:app --reload
goto end

:mlflow_ui
%PY% -m mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
goto end

:docker_build
docker build -t diabetes-api .
goto end

:docker_run
docker run -p 8000:8000 diabetes-api
goto end

:help
echo.
echo Commands:
echo   tasks train          Train and compare models, export the winner to model/
echo   tasks test           Run the pytest suite
echo   tasks serve          Start the API locally on port 8000 (auto-reload)
echo   tasks mlflow-ui      Open the MLflow UI on port 5001
echo   tasks docker-build   Build the diabetes-api Docker image
echo   tasks docker-run     Run the container on port 8000
echo.

:end