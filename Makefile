.PHONY: install test lint format train predict docker-build

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	flake8 . --max-line-length=100 --exclude=.git,__pycache__
	black --check --diff .
	isort --check-only .

format:
	black .
	isort .

extract:
	python extract_features.py

train:
	python train_model.py

tune:
	python tune_model.py

evaluate:
	python evaluate.py --output-dir evaluation/

docker-build:
	docker build -t deepfake-audio-detection:latest .

docker-up:
	docker compose up --build
