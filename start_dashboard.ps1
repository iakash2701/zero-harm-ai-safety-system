Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run dashboard/app.py --server.port 8501 --server.headless true
