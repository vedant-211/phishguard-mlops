import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            fitted_models = {}
            for name, model in models.items():
                param_grid = params.get(name, {})
                logging.info(f"Tuning {name}...")
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="f1")
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                y_pred = best_model.predict(X_test)
                score = f1_score(y_test, y_pred)
                report[name] = float(score)
                fitted_models[name] = best_model
            return report, fitted_models
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42)
            }

            params = {
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [10, 15]},
                "Decision Tree": {"max_depth": [5, 10]},
                "Gradient Boosting": {"n_estimators": [50], "learning_rate": [0.1]},
                "Logistic Regression": {"C": [0.1, 1.0]},
                "AdaBoost": {"n_estimators": [50]}
            }

            model_report, fitted_models = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_model_name = max(model_report, key=model_report.get)
            best_model = fitted_models[best_model_name]

            # Metric calculations
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            test_metric = ClassificationMetricArtifact(
                f1_score=float(f1_score(y_test, y_test_pred)),
                precision_score=float(precision_score(y_test, y_test_pred)),
                recall_score=float(recall_score(y_test, y_test_pred))
            )
            train_metric = ClassificationMetricArtifact(
                f1_score=float(f1_score(y_train, y_train_pred)),
                precision_score=float(precision_score(y_train, y_train_pred)),
                recall_score=float(recall_score(y_train, y_train_pred))
            )

            # MLflow logging
            import mlflow

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            with mlflow.start_run():
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("f1_score", test_metric.f1_score)
                mlflow.log_metric("precision", test_metric.precision_score)
                mlflow.log_metric("recall", test_metric.recall_score)
                mlflow.sklearn.log_model(best_model, "model")

            # Save models
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

            os.makedirs("final_model", exist_ok=True)
            joblib.dump(best_model, os.path.join("final_model", "model.pkl"))

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
                best_model_name=best_model_name,
                models_report=model_report
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
