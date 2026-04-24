# Contributing to Deepfake Audio Detection

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/deepfake-audio-detection.git`
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pytest pytest-cov soundfile flake8 black isort
   ```

## Development Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes and add tests under `tests/`
3. Run the full test suite: `pytest tests/ -v`
4. Run linters: `flake8 . && black --check . && isort --check-only .`
5. Commit following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `refactor:` code restructuring without behaviour change
   - `test:` adding or updating tests
   - `docs:` documentation only
   - `chore:` maintenance tasks (deps, CI, etc.)
   - `build:` Docker / packaging
6. Open a pull request against `main`

## Code Style

- Line length: **100 characters** (Black + flake8)
- Import order: **isort** with `profile = "black"`
- Type hints encouraged for all public functions
- Docstrings required for all public functions and classes

## Testing

- Every new feature or bug-fix must include tests
- Aim for >= 80 % coverage on new code
- Use `pytest` fixtures and `monkeypatch` rather than patching globals

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS
