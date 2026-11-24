@echo off
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || (echo Python 3.10 or 3.11 is needed: python.org & pause & exit /b)
  call .venv\Scripts\activate
  pip install -r requirements.txt
  pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
) else (
  call .venv\Scripts\activate
)
python app.py
pause
