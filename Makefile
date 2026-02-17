.PHONY: install run test lint docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	streamlit run src/ui/app.py --server.port=8501

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

docker-build:
	docker compose build

docker-run:
	docker compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage
