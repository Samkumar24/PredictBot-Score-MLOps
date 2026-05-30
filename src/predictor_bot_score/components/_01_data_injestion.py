from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.entity import Data_injestion_config
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.utils.common import yaml_load , create_directories
from dataclasses import dataclass
from pathlib import Path
import os
import boto3
from botocore.exceptions import (
    ClientError
)
from dotenv import load_dotenv

load_dotenv()  # reads your .env file



class Data_injestion:

    def __init__(self, config : Data_injestion_config):

        self.config = config
        self.s3 =   boto3.client(
                        's3', 
                        aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
        self.BUCKET_NAME = self.config.bucket_name
        self.FILE_NAME = self.config.file_name

    def connection_check(self):

        try:

            response = self.s3.list_objects_v2(Bucket=self.BUCKET_NAME)
            return True

        except ClientError as c:
            # Extract the specific AWS error code dictionary string
            error_code = c.response['Error']['Message']
            print( error_code)
            raise
            
    
    def file_check(self):

        try:
            if self.connection_check() :
                logger.info("S3 . Connection exists ")
                response  = self.s3.list_objects_v2(Bucket = self.BUCKET_NAME)
                FIlES =([i['Key'] for i in response['Contents']])

                if self.FILE_NAME in FIlES:
                    logger.info(f"{self.FILE_NAME} FILE EXISTS")
                    return self.FILE_NAME
                else: 
                    raise FileNotFoundError(
                        f"'{self.FILE_NAME}' does not exist in bucket '{self.BUCKET_NAME}'")


        except ClientError as e:
            logger.error(f"S3 connection error: {e.response['Error']['Message']}")
            raise

    def s3_downlaod_data(self):
    
        try:
            s3_file_name =  self.file_check() 
            file_name = os.path.basename(s3_file_name)
            save_path = os.path.join(self.config.raw_data,file_name)

            if os.path.exists(save_path):
                logger.info(f"{save_path}: {self.FILE_NAME}Path exists ")

            else:
                content = self.s3.download_file(
                        Bucket=self.BUCKET_NAME,
                        Key=self.FILE_NAME,
                        Filename=str(save_path)
                    )
                logger.info(f"{self.FILE_NAME} created at path {save_path}")

        except FileNotFoundError as e:
            logger.error(f"Download failed — file not found: {e}")
            raise
        except ClientError as e:
            logger.error(f"Download failed — AWS error: {e.response['Error']['Message']}")
            raise


            


