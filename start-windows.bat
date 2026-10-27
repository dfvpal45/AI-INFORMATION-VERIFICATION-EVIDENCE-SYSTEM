@echo off
cd /d "%~dp0"
echo Start Docker Desktop first, then run docker compose up --build.
docker compose up --build
pause
