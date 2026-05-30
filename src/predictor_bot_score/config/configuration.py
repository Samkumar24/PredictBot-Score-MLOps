from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.utils.common import yaml_load , create_directories
from dataclasses import dataclass
from pathlib import Path
import os
import boto3
from src.predictor_bot_score.entity import Data_injestion_config

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
