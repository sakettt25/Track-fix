# Track-fix - Streamlit App Launcher (PowerShell)

Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Cyan
Write-Host "Open your browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host ""

streamlit run app.py
