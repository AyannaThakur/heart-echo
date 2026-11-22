@echo off
cd /d "%~dp0"
if not exist models\cardiolens_asd.npz python train.py
python -m streamlit run app.py
