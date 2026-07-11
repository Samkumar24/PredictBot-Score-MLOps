from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.utils.common import yaml_load , create_directories
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from src.predictor_bot_score.entity import Data_injestion_config
import os
import boto3
import io
import json 
import pandas as pd
from pathlib import Path
from botocore.exceptions import  ClientError
from src.predictor_bot_score.utils.src_util_s3_  import *


from dotenv import load_dotenv

load_dotenv()  # reads your .env file

class Data_injestion:

    def __init__(self, config : Data_injestion_config):

        self.config = config
        self.s3 =  s3_login()
        self.BUCKET_NAME = self.config.bucket_name
        self.FILE_NAME = self.config.file_name
        self.pipeline_run_id = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def connection_check(self):

        try:

            response = self.s3.list_objects_v2(Bucket=self.BUCKET_NAME)
            return True

        except ClientError as c:
            # Extract the specific AWS error code dictionary string
            error_code = c.response['Error']['Message']
            print( error_code)
            raise
            
    
    def lambda_file_check(self):

        try:
            file_keys = []
            if self.connection_check() :
                logger.info("S3 . Connection exists ")
                response  = self.s3.list_objects_v2(Bucket = self.BUCKET_NAME,Prefix='lambda-api-data')

                if 'Contents' in response:
                    for obj in response.get('Contents',[]):
                        if obj['Key'].endswith(".csv"):
                            file_keys.append(obj['Key'])
                            logger.info(f"Found {len(file_keys)} files in 'lambda-api-data/'.")
                        else:
                            logger.warning("No files found under the prefix 'lambda-api-data/'.")
            return file_keys

        except ClientError as e:
            logger.error(f"S3 connection error: {e.response['Error']['Message']}")
            raise

    def raw_data_file_check(self):

        try:
            if self.connection_check() :
                logger.info("S3 . Connection exists ")
                response  = self.s3.list_objects_v2(Bucket = self.BUCKET_NAME,Prefix='raw-master-data')

                if 'Contents' in response:
                    all_keys = [i['Key'] for i in response['Contents']]

                    if self.FILE_NAME in all_keys:
                        logger.info(f"File {self.FILE_NAME} found in raw-master-data.")
                        
                        raw_data_obj = self.s3.get_object(
                            Bucket=self.BUCKET_NAME, 
                            Key=self.FILE_NAME)
                        raw_df = pd.read_csv((raw_data_obj['Body']))
                    
            return raw_df
        except ClientError as e:
            logger.error(f"S3 connection error: {e.response['Error']['Message']}")
            raise

    def lambda_conmbine_files(self,key):
        try:
            obj = self.s3.get_object(Bucket='predict-bot-mlops',Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            df = df.rename(columns={'value':'bot_score'})

            print(f"Read {len(df)} rows from {key}")

            return df
        except ClientError as e:
            print(f"Failed to read {key}: {e.response['Error']['Message']}")
            raise
    
    def combine_files(self ,keys):
        try:
            dfs = []
            for key in keys:
                df = self.lambda_conmbine_files(key)
                if df is not None:
                    dfs.append(df)

            if not dfs:
                logger.warning("No dataframes to combine.")
                return None
            
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Successfully combined {len(dfs)} files.")

            logger.info(f"Successfully combined {len(dfs)} files. ")
            return combined

        except Exception as e:
            logger.error(f"Error during file combination: {e}")
            raise
      

    def run(self):

        try:
            logger.info("=" * 50)
            logger.info("DATA INGESTION PIPELINE STARTED")
            logger.info(f"Run ID : {self.pipeline_run_id}")
            logger.info("=" * 50)

            self.connection_check()

            raw_df  = self.raw_data_file_check()

            lambda_keys  = self.lambda_file_check()
            lamdat_combined_df = self.combine_files(lambda_keys)

            if raw_df is None and lamdat_combined_df is None:
                    raise FileNotFoundError(
                        "No data found — upload master data to raw-master-data/ "
                        "and ensure Lambda has run at least once"
                    )
            
            combined_df = pd.concat([raw_df , lamdat_combined_df],ignore_index=True)

            output_key = f"combined_data/run__{self.pipeline_run_id}/combined_df.csv"

            Manifest = save_manifest(pipline_id=self.pipeline_run_id,output_key=output_key)

            save_file_s3(df=combined_df ,output_key=output_key ,BUCKET_NAME=self.BUCKET_NAME,s3_client=self.s3)

            self_s3_mainfest(manifest=Manifest ,output_key=output_key,BUCKET_NAME=self.BUCKET_NAME,s3_client=self.s3)

            save_local(df=combined_df,local_dir=self.config.ingested_data,pipeline_run_id=self.pipeline_run_id)  ## df, local_dir

            self_local_save_mainfest(manifest=Manifest,local_dir=self.config.ingested_data,pipeline_run_id=self.pipeline_run_id)
            


            logger.info("=" * 50)
            logger.info(f"DATA INGESTION COMPLETE")
            logger.info(f"Total rows : {len(combined_df)}")
            logger.info(f"Output     : s3://{self.BUCKET_NAME}/{output_key}")
            logger.info("=" * 50)

            
            return None
        except FileNotFoundError as e:
            logger.error(f"Ingestion failed — missing data: {e}")
            raise
        except ClientError as e:
            logger.error(f"Ingestion failed — S3 error: {e.response['Error']['Message']}")
            raise
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise


    



            





    



            




    


            


