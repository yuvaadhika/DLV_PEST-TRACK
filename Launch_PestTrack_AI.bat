@echo off
cd /d "%~dp0"
title PestTrack AI - Starting Application
echo Starting PestTrack AI Desktop Application...
start "" ".\pest_env\Scripts\pythonw.exe" "app.py"
exit
