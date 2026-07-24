from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.utils.common import yaml_load , create_directories
from dataclasses import dataclass
from pathlib import Path
import os
import boto3
from src.predictor_bot_score.entity import (Data_injestion_config,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            FeatureEngineeringConfig,
                                            ModelTrainingConfig,
                                            ModelEvaluationConfig)

class yaml_configruation:

    def __init__(self,config_path = CONFIG_PATH):
        self.config_path = yaml_load(config_path)

        create_directories([self.config_path.artifacts_root])

    def get_data_ingestion_config(self)-> Data_injestion_config:

        config_path = self.config_path.data_ingestion

        create_directories([config_path.raw_data,config_path.ingested_data])

        return Data_injestion_config(
            bucket_name = config_path.bucket_name,
            file_name = config_path.file_name,
            raw_data = config_path.raw_data,
            ingested_data = config_path.ingested_data

        )

    def get_data_validation_config(self):

        config = self.config_path.data_validation

        create_directories([config.validated_data , config.quarantine_data])
        
        data_validation_config = DataValidationConfig(
            raw_data_path=Path(config.raw_data_folder),
            validated_data=Path(config.validated_data),
            quarantine_data=Path(config.quarantine_data),
            log_path=Path(config.log_path),
            
            expected_columns=config.expected_columns,
            expected_dtypes=config.expected_dtypes,
            
            missing_thresholds=config.missing_thresholds,
            allow_duplicate_timestamps=config.duplicate_timestamps,
            volume=config.volume,
            
            temporal=config.temporal,
            statistical=config.statistical,
            
            range_checks=config.range_checks,
            outlier_floor=config.outlier_floor,
            max_step_drop=config.max_step_drop,
            
            business_logic=config.business_logic,
            hard_fail_checks=config.hard_fail_checks,
            soft_fail_checks=config.soft_fail_checks,
            bucket_name = self.config_path.s3_config.bucket_name
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self):

        config = self.config_path.data_transformation

        create_directories([config.transformed_data])

        data_transformation_config = DataTransformationConfig(
            validated_data_path=Path(config.validated_data_path),
            transformed_data_dir =Path(config.transformed_data),
            bucket_name = self.config_path.s3_config.bucket_name
        )
        
        return data_transformation_config
    
    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:

        config = self.config_path.feature_engineering

        create_directories([
            config.featured_data_dir,
            config.featured_raw_data,
            config.train_data,
            config.validation_data,
            config.test_data
        ])

        return FeatureEngineeringConfig(
                    transformed_data_path = Path(config.transformed_data_path),
                    featured_data_dir     = Path(config.featured_data_dir),
                    featured_raw_data     = Path(config.featured_raw_data),
                    train_data            = Path(config.train_data),
                    val_data              = Path(config.validation_data),
                    test_data             = Path(config.test_data),
                    features              = list(config.features),
                    lag_columns           = dict(config.lag_columns),
                    rolling               = dict(config.rolling),
                    target_column         = config.target_column,
                    bucket_name =       self.config_path.s3_config.bucket_name
                )

        
    def get_model_training_config(self) -> ModelTrainingConfig:

        config = self.config_path.model_training

        create_directories([config.model_dir])

        return ModelTrainingConfig(
        train_data_path       = Path(config.train_data_path),
        val_data_path         = Path(config.val_data_path),
        model_dir             = Path(config.model_dir),
        active_model_strategy = config.active_model_strategy,
        models                = dict(config.models),
        features              = list(config.features),
        target_column         = config.target_column,
        baseline_mae          = float(config.baseline_mae),
        promotion_criteria    = dict(config.promotion_criteria),
        mlflow_experiment     = config.mlflow.experiment_name,
        mlflow_tracking_uri   = config.mlflow.tracking_uri,
        bucket_name =       self.config_path.s3_config.bucket_name
    )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config_path.model_evaluation
        training_config = self.config_path.model_training.models

        create_directories([config.model_eval_dir, config.model_eval_results])
        
        return ModelEvaluationConfig(
        test_data_path        = Path(config.test_data_path),
        validation_data_path = Path(config.validation_data_path),
        model_dir             = Path(config.model_dir),
        model_eval_dir        = Path(config.model_eval_dir),
        champion_path         = Path(config.champion_path),
        model_eval_results    = Path(config.model_eval_results),
        features              = list(config.features),
        target_column         = config.target_column,
        baseline_feature      = config.baseline_feature,
        baseline_mae          = float(config.baseline_mae),
        improvement_threshold = float(config.improvement_threshold),
        champion_threshold    = float(config.champion_threshold),
        spike_threshold       = float(config.spike_threshold),
        mlflow_experiment     = config.mlflow.experiment_name,
        mlflow_tracking_uri   = config.mlflow.tracking_uri,
        model_training        = dict(training_config),
        bucket_name           = self.config_path.s3_config.bucket_name,          # new
        models                = dict(training_config)
    )