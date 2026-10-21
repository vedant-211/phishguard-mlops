import os
import io
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Network Security – Phishing Website Detection MLOps API",
    description="College DevOps Mini Project - Experiment 10 by Vedant Sham Ghadi (Roll No 11, TE-AI&DS)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_state = {
    "is_trained": False,
    "best_model": None,
    "f1_score": None,
    "precision": None,
    "recall": None,
    "models_report": {},
    "last_run": None
}

@app.on_event("startup")
def startup_event():
    # Load the trained model if the artifacts exist; do not train during startup.
    if os.path.exists("final_model/model.pkl") and os.path.exists("final_model/preprocessor.pkl"):
        pipeline_state["is_trained"] = True
        metrics_path = "final_model/metrics.json"
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r") as metrics_file:
                metrics = json.load(metrics_file)
            pipeline_state["best_model"] = metrics.get("best_model")
            pipeline_state["f1_score"] = metrics.get("f1_score")
            pipeline_state["precision"] = metrics.get("precision")
            pipeline_state["recall"] = metrics.get("recall")
            pipeline_state["models_report"] = metrics.get("models_report", {})
        pipeline_state["last_run"] = datetime.now().isoformat()
    else:
        print("Model artifacts not found; skipping startup training.")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Network Security – Phishing Website Detection</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; text-align: center; }
          .card { background: #1e293b; border-radius: 12px; max-width: 650px; margin: 40px auto; padding: 32px; border: 1px solid #334155; }
          .badge { display: inline-block; background: #0284c7; color: white; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; margin-bottom: 12px; }
          a { color: #38bdf8; text-decoration: none; font-weight: 500; }
          a:hover { text-decoration: underline; }
          .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 20px; border-radius: 8px; margin: 8px; text-decoration: none; }
        </style>
      </head>
      <body>
        <div class="card">
          <div class="badge">Experiment 10 • TE-AI&DS</div>
          <h2>Network Security – Phishing Website Detection</h2>
          <p><strong>Student:</strong> Vedant Sham Ghadi | <strong>Roll No:</strong> 11</p>
          <p style="color: #94a3b8;">End-to-End MLOps Pipeline with FastAPI, Scikit-Learn, Docker & AWS CI/CD</p>
          <div style="margin-top: 24px;">
            <a class="btn" href="/docs">Swagger API Docs</a>
            <a class="btn" href="/api/status">Pipeline Status</a>
            <a class="btn" href="/train">Run /train</a>
          </div>
        </div>
      </body>
    </html>
    """

@app.get("/api/status")
def get_status():
    return JSONResponse(pipeline_state)

@app.get("/train")
def train_pipeline():
    try:
        from networksecurity.pipeline.training_pipeline import TrainingPipeline

        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        pipeline_state["is_trained"] = True
        pipeline_state["best_model"] = artifact.best_model_name
        pipeline_state["f1_score"] = artifact.test_metric_artifact.f1_score
        pipeline_state["precision"] = artifact.test_metric_artifact.precision_score
        pipeline_state["recall"] = artifact.test_metric_artifact.recall_score
        pipeline_state["models_report"] = artifact.models_report
        pipeline_state["last_run"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "message": "Pipeline completed successfully",
            "best_model": artifact.best_model_name,
            "metrics": {
                "f1_score": artifact.test_metric_artifact.f1_score,
                "precision": artifact.test_metric_artifact.precision_score,
                "recall": artifact.test_metric_artifact.recall_score
            },
            "models_comparison": artifact.models_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not os.path.exists("final_model/model.pkl") or not os.path.exists("final_model/preprocessor.pkl"):
            raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        preprocessor = joblib.load("final_model/preprocessor.pkl")
        model = joblib.load("final_model/model.pkl")

        # Drop result column if present
        features = df.drop(columns=["Result"], errors="ignore")
        transformed = preprocessor.transform(features)
        preds = model.predict(transformed)

        df["predicted_column"] = preds
        df["verdict"] = ["Legitimate" if p == 1 else "Phishing" for p in preds]

        return {
            "status": "success",
            "total_samples": len(df),
            "phishing_count": int((preds == 0).sum()),
            "legitimate_count": int((preds == 1).sum()),
            "data": df.head(100).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
