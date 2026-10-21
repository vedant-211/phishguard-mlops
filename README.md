# Network Security – Phishing Website Detection (End-to-End MLOps Pipeline)

College mini project for Agile Software Development & DevOps Lab, Experiment 10. Student: Vedant Sham Ghadi, Roll No. 11, TE-AI&DS.

## Architecture
FastAPI serves the demo/API. The modular `networksecurity` package runs data ingestion, schema/drift validation, KNN imputation, and model training. Training artifacts are saved under `Artifacts/` and the selected model and preprocessor under `final_model/`. MLflow logs locally to `./mlruns` unless `MLFLOW_TRACKING_URI` is configured. Data uses MongoDB Atlas when `MONGO_DB_URL` is set, otherwise `data/phisingData.csv`.

## Run locally or on Replit
Use Python 3.10. Install requirements and start the service on port 8000:

```bash
pip install -r requirements.txt
python app.py
```

The app binds to `0.0.0.0:8000`. Put the provided dataset at `data/phisingData.csv`. Configure `MONGO_DB_URL` only when using MongoDB Atlas. If no dataset is present, the ingestion component generates a synthetic dataset.

## Endpoints
- `GET /` demo landing page
- `GET /docs` Swagger UI
- `GET /train` run the training pipeline
- `POST /predict` upload CSV website-feature rows
- `GET /api/status` current model status

## Docker
```bash
docker build -t phishguard-mlops .
docker run --rm -p 8000:8000 phishguard-mlops
```

## CI/CD and cloud setup
`.github/workflows/main.yml` contains CI, ECR image publishing, and deployment to a self-hosted EC2 runner. Add AWS access keys, region, ECR registry/repository, MongoDB URL and MLflow tracking settings as GitHub Actions repository secrets before enabling deployment. The workflow does not include credentials. ECR, EC2, optional S3, and DagsHub require your own AWS/DagsHub configuration; no cloud resources are provisioned by this repository. The optional S3 sync is not included in this starter implementation.

## Notes
- `data/phisingData.csv` is intentionally not committed; provide the real dataset locally or use synthetic fallback data.
- MongoDB upload helper: `python push_data.py` (requires `MONGO_DB_URL`).
- Generated scores depend on the dataset and training run; dashboard/demo figures should not be treated as measured results unless verified from a run.
