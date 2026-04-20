# ─────────────────────────────────────────────────────────────
# Makefile — Telecom Churn Geospatial Project
# Usage: make <target>
# ─────────────────────────────────────────────────────────────

.PHONY: help install setup data features train evaluate geo-risk predict \
        api dashboard mlflow test test-cov clean docker-up docker-down lint

PYTHON  := python
PIP     := pip
PROJECT := telecom-churn-geospatial

# ── Default target ────────────────────────────────────────────
help:
	@echo ""
	@echo "  $(PROJECT)"
	@echo "  ──────────────────────────────────────────────────"
	@echo "  Setup"
	@echo "    make install       Install all Python dependencies"
	@echo "    make setup         Full first-time project setup"
	@echo ""
	@echo "  Pipeline (run in order)"
	@echo "    make data          Generate synthetic subscriber data"
	@echo "    make validate      Validate raw data quality"
	@echo "    make features      Run feature engineering pipeline"
	@echo "    make train         Train model + log to MLflow"
	@echo "    make evaluate      Run full model evaluation"
	@echo "    make geo-risk      Generate H3 geospatial risk map"
	@echo "    make predict       Score all subscribers (batch)"
	@echo "    make pipeline      Run all steps end-to-end"
	@echo ""
	@echo "  Optional data ingestion"
	@echo "    make opencellid    Download real OpenCelliD tower data"
	@echo "    make osm           Download OpenStreetMap POIs for Jakarta"
	@echo ""
	@echo "  Serving"
	@echo "    make api           Start FastAPI prediction server"
	@echo "    make dashboard     Start Streamlit dashboard"
	@echo "    make mlflow        Start MLflow tracking server"
	@echo ""
	@echo "  Notebooks"
	@echo "    make notebook      Launch Jupyter Lab"
	@echo "    make run-notebooks Run all notebooks as scripts"
	@echo ""
	@echo "  Quality"
	@echo "    make test          Run pytest test suite"
	@echo "    make test-cov      Run tests with coverage report"
	@echo "    make lint          Run flake8 + black check"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-up     Start full stack (API + Dashboard + MLflow)"
	@echo "    make docker-down   Stop all containers"
	@echo ""
	@echo "  make clean          Remove generated data and model artifacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup: install
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/external data/models logs docs
	@echo "✅ Setup complete. Edit .env then run: make pipeline"

# ── Data Pipeline ─────────────────────────────────────────────
data:
	$(PYTHON) src/data_engineering/generate_data.py \
		--n_subscribers 50000 \
		--output data/raw/

validate:
	$(PYTHON) src/data_engineering/data_validation.py \
		--input data/raw/subscribers.parquet

features:
	$(PYTHON) src/features/feature_pipeline.py \
		--input data/raw \
		--output data/processed

train:
	$(PYTHON) src/models/train.py \
		--model xgboost \
		--config configs/config.yaml

train-lgbm:
	$(PYTHON) src/models/train.py \
		--model lightgbm \
		--config configs/config.yaml

train-tune:
	$(PYTHON) src/models/train.py \
		--model xgboost \
		--tune \
		--config configs/config.yaml

evaluate:
	$(PYTHON) src/models/evaluate.py \
		--model_path data/models/churn_model_xgboost.pkl

geo-risk:
	$(PYTHON) src/models/geo_risk_map.py \
		--output data/processed/risk_grid.geojson

predict:
	$(PYTHON) src/models/predict.py \
		--input data/processed/features_full.parquet \
		--output data/processed/predictions_latest.parquet \
		--skip_features \
		--export_csv

pipeline: data validate features train evaluate geo-risk predict
	@echo ""
	@echo "✅ Full pipeline complete!"
	@echo "   Next steps:"
	@echo "     make dashboard    → View results"
	@echo "     make api          → Start prediction API"
	@echo "     make mlflow       → View experiment tracking"

# ── Optional real data ingestion ──────────────────────────────
opencellid:
	@echo "Downloading OpenCelliD tower data for Indonesia..."
	$(PYTHON) src/data_engineering/ingest_opencellid.py \
		--country ID \
		--output data/external
	@echo "✅ OpenCelliD data downloaded to data/external/"

osm:
	@echo "Downloading OpenStreetMap POIs for Jakarta..."
	$(PYTHON) src/data_engineering/ingest_osm.py \
		--city "Jakarta" \
		--output data/external
	@echo "✅ OSM POIs downloaded to data/external/"

# ── Serving ───────────────────────────────────────────────────
api:
	@echo "Starting FastAPI server at http://localhost:8000"
	@echo "Swagger docs: http://localhost:8000/docs"
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

dashboard:
	@echo "Starting Streamlit dashboard at http://localhost:8501"
	streamlit run dashboards/streamlit_app.py \
		--server.port 8501 \
		--server.address 0.0.0.0

mlflow:
	@echo "Starting MLflow UI at http://localhost:5000"
	mlflow ui --host 0.0.0.0 --port 5000

# ── Notebooks ─────────────────────────────────────────────────
notebook:
	jupyter lab notebooks/

run-notebooks:
	@echo "Running notebooks as scripts..."
	$(PYTHON) notebooks/01_data_exploration.py
	$(PYTHON) notebooks/02_feature_engineering.py
	$(PYTHON) notebooks/03_model_training.py
	$(PYTHON) notebooks/04_geospatial_analysis.py
	@echo "✅ All notebooks complete. Charts saved to docs/"

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -v --tb=short -x -q  # stop on first failure

# ── Code quality ──────────────────────────────────────────────
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503 || true
	black src/ --check --line-length=100 || true

format:
	black src/ --line-length=100
	isort src/ --profile=black

# ── Docker ────────────────────────────────────────────────────
docker-up:
	cd mlops/docker && docker-compose up --build -d
	@echo ""
	@echo "✅ Services started:"
	@echo "   API       : http://localhost:8000"
	@echo "   Dashboard : http://localhost:8501"
	@echo "   MLflow    : http://localhost:5000"
	@echo "   Airflow   : http://localhost:8080"

docker-down:
	cd mlops/docker && docker-compose down
	@echo "✅ All containers stopped."

docker-logs:
	cd mlops/docker && docker-compose logs -f

docker-rebuild:
	cd mlops/docker && docker-compose down && docker-compose up --build

# ── Cleanup ───────────────────────────────────────────────────
clean:
	@echo "Removing generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/
	@echo "✅ Cleaned Python cache files."

clean-data:
	@echo "⚠️  Removing all generated data and models..."
	rm -f data/raw/*.parquet data/raw/*.csv
	rm -f data/processed/*.parquet data/processed/*.geojson data/processed/*.html
	rm -f data/models/*.pkl data/models/*.png data/models/*.csv data/models/*.txt
	@echo "✅ Data files removed. Run: make pipeline to regenerate."

clean-all: clean clean-data
	rm -rf mlruns/ logs/ mlflow.db
	@echo "✅ Full clean complete."
