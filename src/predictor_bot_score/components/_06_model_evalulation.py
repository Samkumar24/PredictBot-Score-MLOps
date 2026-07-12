from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.constants import CONFIG_PATH
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass
from src.predictor_bot_score.utils.model_factory import get_model , get_fit_kwargs , get_mlflow_logger , MODEL_REGISTRY
from src.predictor_bot_score.entity import ModelEvaluationConfig
import glob
from datetime import datetime
import mlflow
import shutil
import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path
import time
import gc
import os
from src.predictor_bot_score.utils.src_util_s3_ import *

class Model_evalulation:

    def __init__(self,config : ModelEvaluationConfig):
        self.config = config
        self.s3 = s3_login()
        self.BUCKET_NAME = self.config.bucket_name
        self.test_df = self._read_data(prefix= "test")
        self.val_df  = self._read_data(prefix= "val")

    def _read_data(self,  prefix):
        try:
            keys = []
            paginator = self.s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.BUCKET_NAME, Prefix='feature_engineering'):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if prefix in key and key.endswith('.csv') and obj['Size'] > 0:
                        keys.append(key)

            if not keys:
                raise FileNotFoundError(
                    f"No {prefix} CSV found under s3://{self.BUCKET_NAME}/feature_engineering"
                )
            lastest_file_key = sorted(keys)[-1]
            
            obj = self.s3.get_object(Bucket=self.BUCKET_NAME, Key=lastest_file_key)
            df  = pd.read_csv(io.BytesIO(obj['Body'].read()))
            return df
            

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        

    def _latest_models(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            model_keys = []
            for page in paginator.paginate(Bucket=self.BUCKET_NAME, Prefix='model_building'):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.pkl') and obj['Size'] > 0:
                        model_keys.append(key)

            if not model_keys:
                raise FileNotFoundError(
                    f"No model pkl files found under s3://{self.BUCKET_NAME}/model_building"
                )     
            run_folders = {}
            for key in model_keys:
                folder = '/'.join(key.split('/')[:2])   # e.g. model_building/model_trained__2026_07_11_15
                run_folders.setdefault(folder, []).append(key)

                latest_folder = sorted(run_folders.keys())[-1]
                model_keys    = run_folders[latest_folder]

            logger.info(f"Loading models from: {latest_folder}")
            logger.info(f"Found {len(model_keys)} model files")
            
            return model_keys     
        except FileNotFoundError as e:
                logger.error(f"Model file not found: {e}")
                raise

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise       
    
    def _prepare_data(self ,df):

        try:
            X = df[self.config.features]
            y = df[self.config.target_column]

            return X ,y
        except Exception as e:
            raise
    
    def _smape(self, actual, predicted):
        try:
            return float(
                100 * np.mean(
                    2 * np.abs(predicted - actual) /
                    (np.abs(actual) + np.abs(predicted) + 1e-8)
                )
            )
        except Exception as e:
            logger.error(f"SMAPE calculation failed: {str(e)}")
            raise

    def _evaluate_baseline(self):

        try:
            logger.info("")
            logger.info("STEP 1 - BASELINE EVALUATION")
            logger.info("-" * 50)
            
            X , y = self._prepare_data(self.test_df)
            baseline_pred = X[self.config.baseline_feature]

            baseline_mae  = float(mean_absolute_error(y, baseline_pred))

            logger.info(f"Baseline MAE : {baseline_mae:.6f}")
            logger.info("PASSED - Baseline evaluated")

            return baseline_mae

        except Exception as e:
            logger.error(f"Baseline evaluation failed: {str(e)}")
            raise

            
        except Exception as e:
            raise

    
    def _deployment_decision(self , baseline_mae):

        try:
            logger.info("")
            logger.info("STEP 3 - DEPLOYMENT DECISION")
            logger.info("-" * 50)
            
            passing = {}
            model_eval_files = glob.glob(os.path.join(self.config.model_eval_results , "*json"))
            model_result =  max(model_eval_files ,key=os.path.getmtime)
            
            with open(model_result ,'r') as f:
                data = json.load(f)
                #return data
            
            for model_name , metrics in data.items():
                test_mae = metrics['metrics']['test_mae']
                improvement = (baseline_mae - test_mae) / baseline_mae * 100

                logger.info(f"Evaluating {model_name}: MAE={test_mae:.6f}, Improvement={improvement:.2f}")

                logger.info(f"Baseline MAE  : {baseline_mae:.6f}")
                logger.info(f"Model MAE     : {test_mae:.6f}")
                logger.info(f"Improvement   : {improvement:.2f}%")
                logger.info(f"Threshold     : {self.config.improvement_threshold * 100:.0f}%")

                if improvement >= self.config.improvement_threshold * 100:
                    passing[model_name] = improvement

            if not passing:
                logger.error("FAILED - No model beat the baseline threshold")
                logger.error("Nothing deployed; current champion stays in production")
                return False, None, None
            
            best_model_name = max(passing ,key=passing.get)
            logger.info(f"PASSED - {best_model_name} beats threshold (improvement: {passing[best_model_name]:.2f}%)")
            return True, best_model_name, passing[best_model_name]
                
        except Exception as e:
            raise

    def log_to_mlflow(self):
        try:
            logger.info("")
            logger.info("STEP 2 - EVALUATE + LOG TO MLFLOW")
            logger.info("-" * 50)
            self.model = self._latest_models()

            X_test, y_test = self._prepare_data(self.test_df)
            X_val, y_val = self._prepare_data(self.val_df)

            mlflow.end_run() 
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)

            report = {}
            for model_path in self.model:
                model_name = os.path.basename(model_path).replace(".pkl", "")
                
                obj   = self.s3.get_object(Bucket=self.BUCKET_NAME, Key=model_path)
                model = pickle.loads(obj['Body'].read())
                logger.info(f"Loaded model from S3: {model_path}")
                
                val_pred = model.predict(X_val)
                val_mae = float(mean_absolute_error(y_val, val_pred))
                val_smape = self._smape(y_val.values, val_pred)

                test_pred = model.predict(X_test)
                test_mae = float(mean_absolute_error(y_test, test_pred))
                test_smape = self._smape(y_test.values, test_pred)

                metrics = {
                        "val_mae": float(mean_absolute_error(y_val, val_pred)),
                        "val_smape": float(self._smape(y_val.values, val_pred)),
                        "test_mae": float(mean_absolute_error(y_test, test_pred)),
                        "test_smape": float(self._smape(y_test.values, test_pred))
                    }
                
        
                with mlflow.start_run(run_name=model_name):

                    params = self.config.model_training[model_name]['params']
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("evaluated_at", datetime.now().isoformat())

                    log_fn = get_mlflow_logger(model_name)
                    log_fn(model, artifact_path="model")

                    report[model_name] = {
                            "params": params,
                            "metrics": metrics,
                            "timestamp": datetime.now().isoformat()
                        }
                    logger.info(f"{model_name} -- val_mae={val_mae:.6f}, test_mae={test_mae:.6f}")
                    logger.info("PASSED - logged to MLflow")
                    logger.info("-" * 50)

            timestamp = datetime.now().strftime("%Y_%m_%d_%H")
            report_name = f"Model_evaluation_report_{timestamp}.json"

            report_path = os.path.join(self.config.model_eval_results, report_name)
            with open(report_path, "w") as f:
                    json.dump(report, f, indent=4)
                        
                        # Log report as an artifact in MLflow
            #mlflow.log_artifact(report_path)

            s3_key = f"model_evaluation/{report_name}"
            self.s3.put_object(
                Bucket      = self.BUCKET_NAME,
                Key         = s3_key,
                Body        = json.dumps(report, indent=4),
                ContentType = 'application/json'
            )
            logger.info(f"Report saved to S3 -> s3://{self.BUCKET_NAME}/{s3_key}")
            logger.info("PASSED - MLflow logging and report generation complete")
                        
            logger.info(f"Report saved {report_path}")
            logger.info("PASSED - MLflow logging and report generation complete")
            return report

        except Exception as e:
                logger.error(f"Failed to log to MLflow: {str(e)}")
                raise
    
    def run(self):
        try:
            logger.info("=" * 50)
            logger.info("MODEL EVALUATION PIPELINE STARTED")
            logger.info("=" * 50)

            baseline_mae = self._evaluate_baseline()
            self.log_to_mlflow()
            promoted, best_model_name, improvement = self._deployment_decision(baseline_mae)

            logger.info("=" * 50)
            logger.info(f"Best model : {best_model_name if promoted else 'None'}")
            logger.info("MODEL EVALUATION PIPELINE COMPLETE")
            logger.info("=" * 50)
            return promoted, best_model_name, improvement

        except Exception as e:
            logger.error(f"Model evaluation pipeline failed: {str(e)}")
            raise
        finally:
            self.test_df = None
            self.val_df = None
            gc.collect()
            logger.info("Memory cleared")




            
    

            





            
    

            
