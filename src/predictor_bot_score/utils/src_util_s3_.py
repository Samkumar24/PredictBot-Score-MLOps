import os
import boto3
import io
from datetime import datetime
from src.predictor_bot_score.logger import logger
import json
from  botocore.exceptions import (
    ClientError)

def s3_login():
      return boto3.client(
                    's3', 
                        aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

def save_manifest(pipline_id,output_key):
            
                
            manifest = {
                    "pipeline_run_id" : pipline_id,
                    "output_key"      : output_key,
                    "completed_at"    : datetime.now().isoformat(),
                    "status"          : "success",
                }
            return manifest

def save_file_s3( df , output_key, BUCKET_NAME, s3_client):
        try:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            s3_client.put_object(
                Bucket = BUCKET_NAME,
                Key =   output_key,
                Body         = buf.getvalue(),
                ContentType  = 'text/csv'
            )
            logger.info(f"Saved {len(df)} rows  s3://{BUCKET_NAME}/{output_key}")
        except ClientError as e:
            logger.error(f"S3 save failed: {e.response['Error']['Message']}")
            raise 

def self_s3_mainfest(manifest,output_key ,BUCKET_NAME ,s3_client):
        try:
            manifest_key = output_key.rsplit('.', 1)[0] + '.json'
            
            s3_client.put_object(
                Bucket = BUCKET_NAME,
                Key = manifest_key,
                Body        = json.dumps(manifest, indent=2),
                ContentType = 'application/json'
            )
            logger.info(f"Saved (manifest) rows  s3://{BUCKET_NAME}/{output_key}")
        except ClientError as e:
            logger.error(f"S3 save failed: {e.response['Error']['Message']}")
            raise

def self_local_save_mainfest(manifest , local_dir ,pipeline_run_id):    
        try:            
            #manifest_key = f"combined_data/run__{self.pipeline_run_id}/manifest.json"
            local_dir = os.path.join(local_dir,
                                    f"lambda_run__{pipeline_run_id}")
            
            local_path = os.path.join(local_dir,"manifest.json")
            with open(local_path , 'w') as f:
                json.dump(manifest ,f ,indent=4)
                logger.info(f"Manifest saved locally  {local_dir}")
            return None
        except ClientError as e:
            logger.error(f"Manifest S3 save failed: {e.response['Error']['Message']}")
            raise
        except Exception as e:
            logger.error(f"Manifest local save failed: {e}")
            raise

def save_local(df,local_dir ,pipeline_run_id):
        try:
            local_dir = os.path.join(local_dir,
                                    f"lambda_run__{pipeline_run_id}"
        )
            os.makedirs(local_dir ,exist_ok=True)
            local_path = os.path.join(local_dir, "combined.csv")
            df.to_csv(local_path, index=False)

            logger.info(f"Saved locally ****  {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Local save failed: {e}")
            raise