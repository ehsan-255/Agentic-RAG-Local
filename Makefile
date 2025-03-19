.PHONY: setup test clean lint format run run-api

# Setup the project
setup:
	pip install -r requirements.txt
	python scripts/setup_database.py

# Run the tests
test:
	pytest tests/

# Clean up the project
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run linting
lint:
	flake8 src/ tests/

# Format the code
format:
	black src/ tests/

# Run the UI application
run:
	cd src/ui && streamlit run streamlit_app.py
	
# Run the API application
run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload 