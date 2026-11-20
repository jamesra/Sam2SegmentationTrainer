@echo off
REM Build the segmentation server image. The compose file's build context is the Sam2Trainer repository root.
cd /d "%~dp0"
docker compose -f docker-compose.yml build segmentation-server
