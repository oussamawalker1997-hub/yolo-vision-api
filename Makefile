.PHONY: install test lint run docker-build docker-run

install:
	pip install -r requirements.txt
	pip install black flake8 isort pre-commit
	pre-commit install

test:
	pytest tests/

lint:
	black app tests
	isort app tests
	flake8 app tests

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t vision-api:latest .

docker-run:
	docker run -p 8000:8000 vision-api:latest
