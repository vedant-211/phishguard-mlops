import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            imputer = KNNImputer(n_neighbors=3, weights="uniform")
            processor = Pipeline(steps=[("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            input_feature_train_df = train_df.drop(columns=["Result"], axis=1)
            target_feature_train_df = train_df["Result"].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=["Result"], axis=1)
            target_feature_test_df = test_df["Result"].replace(-1, 0)

            preprocessor = self.get_data_transformer_object()
            transformed_input_train = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test, np.array(target_feature_test_df)]

            # Save arrays
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            joblib.dump(preprocessor, self.data_transformation_config.transformed_object_file_path)
            
            # Also sync to final_model directory
            os.makedirs("final_model", exist_ok=True)
            joblib.dump(preprocessor, os.path.join("final_model", "preprocessor.pkl"))

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
