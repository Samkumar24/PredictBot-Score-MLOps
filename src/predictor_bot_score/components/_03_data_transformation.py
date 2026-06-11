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

            
            logger.info("STEP 1 - TYPE CONVERSION")
            logger.info("-" * 50)
            self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], utc=True)
            self.data["bot_score"] = self.data["bot_score"].astype(float)
            logger.info("PASSED - Types converted")

            
            logger.info("STEP 2 - SORT BY TIMESTAMP")
            logger.info("-" * 50)
            self.data = self.data.sort_values("timestamp").reset_index(drop=True)
            logger.info("PASSED - Data sorted")

            
            logger.info("STEP 3 - DROP DUPLICATES")
            logger.info("-" * 50)
            before = len(self.data)
            self.data = self.data.drop_duplicates(subset="timestamp")
            after = len(self.data)
            logger.info(f"PASSED - Removed {before - after} duplicate rows")

            
            logger.info("STEP 4 - DROP NULLS")
            logger.info("-" * 50)
            self.data = self.data.dropna().reset_index(drop=True)
            logger.info(f"PASSED - {len(self.data)} clean rows remaining")

            # save
            #timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            out_path  = os.path.join(
                self.config.transformed_data_dir,
                f"transformed.csv"
            )
            self.data.to_csv(out_path, index=False)
            logger.info(f"Data saved - {out_path}")
            logger.info("=" * 50)
            logger.info("DATA TRANSFORMATION COMPLETE")
            

        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise