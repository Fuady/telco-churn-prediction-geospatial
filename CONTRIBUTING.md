# Contributing

Thank you for your interest in contributing to this project!

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/telecom-churn-geospatial.git
cd telecom-churn-geospatial

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -r requirements.txt
pip install -e .
pip install black flake8 isort pytest-cov
```

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Quick test (stop on first failure)
pytest tests/ -x -q
```

## Code Style

This project uses:
- `black` for formatting (line length: 100)
- `flake8` for linting
- `isort` for import sorting

```bash
make format   # auto-format
make lint     # check only
```

## Project Structure

Please follow the existing module structure when adding code:
- New data sources → `src/data_engineering/`
- New features → `src/features/`
- New models → `src/models/`
- New visualizations → `src/visualization/`
- Tests → `tests/test_{module_name}.py`

## Pull Request Process

1. Fork the repo and create a feature branch: `git checkout -b feature/my-new-feature`
2. Write tests for any new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Run formatting: `make format`
5. Update `docs/` if your change affects architecture or data sources
6. Submit a pull request with a clear description of what changes and why

## Reporting Issues

Please include:
- Python version and OS
- Full error traceback
- Steps to reproduce
- Expected vs actual behaviour
