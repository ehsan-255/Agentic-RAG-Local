@echo off
echo Setting up Agentic RAG Local project...

echo Installing Python dependencies...
pip install -r requirements.txt

echo Setting up database...
python scripts/setup_database.py

echo Configuring PostgreSQL...
call scripts/configure_postgresql.bat

echo Setup complete!
echo.
echo To run the application, use: "cd src/ui && streamlit run streamlit_app.py"
echo. 