import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig,
    DataTransformationConfig, ModelTrainerConfig
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def run_pipeline(self):
        try:
            logging.info("Starting MLOps Training Pipeline...")
            
            # Stage 1: Data Ingestion
            data_ingestion = DataIngestion(DataIngestionConfig(self.training_pipeline_config))
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            # Stage 2: Data Validation
            data_validation = DataValidation(data_ingestion_artifact, DataValidationConfig(self.training_pipeline_config))
            data_validation_artifact = data_validation.initiate_data_validation()

            # Stage 3: Data Transformation
            data_transformation = DataTransformation(data_validation_artifact, DataTransformationConfig(self.training_pipeline_config))
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            # Stage 4: Model Training
            model_trainer = ModelTrainer(ModelTrainerConfig(self.training_pipeline_config), data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("Training pipeline completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
