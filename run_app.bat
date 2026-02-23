@echo off
REM Track-fix - Streamlit App Launcher (Windows)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Starting Streamlit app...
echo Open your browser at http://localhost:8501
echo.

streamlit run app.py
