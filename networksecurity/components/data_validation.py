import os
import sys
import yaml
import pandas as pd
from scipy.stats import ks_2samp
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            with open("data_schema/schema.yaml", "r") as schema_file:
                self.schema_config = yaml.safe_load(schema_file)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = len(self.schema_config["columns"])
            return len(dataframe.columns) == expected_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                is_same_dist = ks_2samp(d1, d2)
                p_val = float(is_same_dist.pvalue)
                has_drift = p_val < threshold
                if has_drift:
                    status = False
                report[col] = {
                    "p_value": p_val,
                    "drift_status": bool(has_drift)
                }
            
            drift_dir = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(drift_dir, exist_ok=True)
            with open(self.data_validation_config.drift_report_file_path, "w") as f:
                yaml.dump(report, f)
            logging.info("Kolmogorov–Smirnov data drift check saved to report.yaml")
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            valid_train = self.validate_number_of_columns(train_df)
            valid_test = self.validate_number_of_columns(test_df)
            validation_status = valid_train and valid_test
            
            self.detect_dataset_drift(train_df, test_df)
            
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
            
            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path="",
                invalid_test_file_path="",
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
