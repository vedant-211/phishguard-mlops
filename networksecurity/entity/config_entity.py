import os
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp: str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = "NetworkSecurity"
        self.artifact_dir: str = os.path.join("Artifacts", self.timestamp)

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, "feature_store", "phisingData.csv")
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, "ingested", "train.csv")
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, "ingested", "test.csv")
        self.train_test_split_ratio: float = 0.2
        self.collection_name: str = "NetworkData"
        self.database_name: str = "AI"

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, "validated")
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, "invalid")
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, "train.csv")
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, "test.csv")
        self.drift_report_file_path: str = os.path.join(self.data_validation_dir, "drift_report", "report.yaml")

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir, "transformed", "train.npy")
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir, "transformed", "test.npy")
        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir, "transformed_object", "preprocessor.pkl")

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, "trained_model", "model.pkl")
        self.expected_accuracy: float = 0.6
        self.overfitting_underfitting_threshold: float = 0.05
