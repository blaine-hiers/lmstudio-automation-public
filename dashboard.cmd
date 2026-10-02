@echo off
rem Afterhours control: launch any task by hand. Pin this to the taskbar or desktop.
cd /d "%~dp0"
.venv\Scripts\python -m tasks.dashboard %*
