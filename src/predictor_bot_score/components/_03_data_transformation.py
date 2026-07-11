from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.entity import DataTransformationConfig
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import pandas as pd 
import os
from src.predictor_bot_score.utils.src_util_s3_  import *

class Data_transformation:

    def __init__(self, config : DataTransformationConfig):
        self.config = config
        self.BUCKET_NAME = self.config.bucket_name
        self.s3 = s3_login()
        self.data = self.read_data()
        self.pipeline_run_id = datetime.now().strftime("%Y_%m_%d_%H")
        

    def read_data(self):
        try:

            keys = []

            logger.info("=" * 50)
            logger.info("DATA TRANSFORMATION PIPELINE STARTED")
            logger.info("=" * 50)
            
            for page in self.s3.list_objects_v2(Bucket=self.BUCKET_NAME,Prefix='data_validation')['Contents']:
                if page.get('Key').endswith('.csv'):
                    keys.append(page.get('Key'))

            logger.info("S3 . Connection exists ")

            combined_file_key = sorted(keys)[-1]

            return combined_file_key
        
        except ClientError as e:

            # This is your custom message
            print("--- ALERT: The file is missing! Please check the path. ---")
            
            # This is the log file entry
            logger.error(f"File not found at: {self.BUCKET_NAME}")
            raise

    def transformed_data(self):
        
        try:
            file_key = self.data
            
            obj = self.s3.get_object(Bucket=self.BUCKET_NAME,Key=file_key)
            transformed_df = pd.read_csv(io.BytesIO(obj['Body'].read()))

            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"], utc=True)

            transformed_df["bot_score"] = transformed_df["bot_score"].astype(float)

            return transformed_df
        
        except Exception as e:
            logger.info(e)
            raise

    def run(self):

        try:

            transformed_df = self.transformed_data()

            output_key = f"data_transformation/run__{self.pipeline_run_id}/transformed_data.csv"

            Manifest = save_manifest(pipline_id=self.pipeline_run_id,output_key=output_key)

            save_file_s3(df=transformed_df ,output_key=output_key ,BUCKET_NAME=self.BUCKET_NAME,s3_client=self.s3)

            self_s3_mainfest(manifest=Manifest ,output_key=output_key,BUCKET_NAME=self.BUCKET_NAME,s3_client=self.s3)

            save_local(df=transformed_df,local_dir=self.config.transformed_data_dir,pipeline_run_id=self.pipeline_run_id)  ## df, local_dir

            self_local_save_mainfest(manifest=Manifest,local_dir=self.config.transformed_data_dir,pipeline_run_id=self.pipeline_run_id)

            logger.info("=" * 50)
            logger.info("DATA TRANSFORMATION PIPELINE COMPLETED")
            logger.info("=" * 50)

        except Exception as e:
            logger.info(e)
            raise
    