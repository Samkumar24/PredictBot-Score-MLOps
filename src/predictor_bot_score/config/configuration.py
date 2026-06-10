from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.utils.common import yaml_load , create_directories
from dataclasses import dataclass
from pathlib import Path
import os
import boto3
from src.predictor_bot_score.entity import (Data_injestion_config,
                                            DataValidationConfig,DataTransformationConfig)

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
            soft_fail_checks=config.soft_fail_checks
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self):

        config = self.config_path.data_transformation

        create_directories([config.transformed_data])

        data_transformation_config = DataTransformationConfig(
            validated_data_path=Path(config.validated_data_path),
            transformed_data_dir =Path(config.transformed_data)
        )
        
        return data_transformation_config


