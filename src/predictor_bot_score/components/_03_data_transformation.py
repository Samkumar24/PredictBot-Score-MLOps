from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.entity import DataTransformationConfig
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import pandas as pd 
import os

class Data_transformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data = self.read_data()
    def read_data(self):
        try:
            logger.info(f"Attempting to read data from: {self.config.validated_data_path}")
            data = pd.read_csv(self.config.validated_data_path)
            logger.info("Data successfully loaded into memory.")
            return data
        
        except FileNotFoundError:
            # This is your custom message
            print("--- ALERT: The file is missing! Please check the path. ---")
            
            # This is the log file entry
            logger.error(f"File not found at: {self.config.validated_data_path}")
            
             

    def transformed_data(self):
        try:
            logger.info("=" * 50)
            logger.info("DATA TRANSFORMATION PIPELINE STARTED")
            logger.info("=" * 50)

            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], utc=True)

            self.data["bot_score"] = self.data["bot_score"].astype(float)

            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            out_path  = self.config.transformed_data_dir / f"_{timestamp}.csv"


            self.data.to_csv(out_path,index=False)
            logger.info(f"Data saved  {out_path}")
            logger.info(f"DATA TRANSFORMATION DONE")

        
        except Exception as e:
            logger.info(e)