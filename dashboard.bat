@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
endlocal

