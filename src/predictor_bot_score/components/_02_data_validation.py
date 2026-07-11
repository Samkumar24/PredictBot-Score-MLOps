from dataclasses import dataclass 
from src.predictor_bot_score.logger import logger
from src.predictor_bot_score.config.configuration import yaml_load , create_directories
from src.predictor_bot_score.constants import CONFIG_PATH
from src.predictor_bot_score.entity import DataValidationConfig
from pathlib import Path
from datetime import datetime
from src.predictor_bot_score.utils.src_util_s3_ import *
import pandas as pd

class Data_validation:

    def __init__(self, config : DataValidationConfig):
        self.config = config
        self.BUCKET_NAME = self.config.bucket_name
        self.s3 = s3_login()
        self.df     = self.read_data()
        self.pipeline_run_id = datetime.now().strftime("%Y_%m_%d_%H")
        
        

    def read_data(self):
        try:

            keys = []

            logger.info("=" * 50)
            logger.info("DATA VALIDATION PIPELINE STARTED")
            logger.info("=" * 50)
            
            for page in self.s3.list_objects_v2(Bucket=self.BUCKET_NAME,Prefix='combined_data')['Contents']:
                if page.get('Key').endswith('.csv'):
                    keys.append(page.get('Key'))

            logger.info("S3 . Connection exists ")

            combined_file_key = sorted(keys)[-1]


            obj = self.s3.get_object(Bucket=self.BUCKET_NAME,Key=combined_file_key)
            transformed_df = pd.read_csv(io.BytesIO(obj['Body'].read()))

            transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"], utc=True)

            transformed_df["bot_score"] = transformed_df["bot_score"].astype(float)

            timestamp = datetime.now().strftime("%Y_%m_%d_%H")
            out_path  = self.config.validated_data / f"_{timestamp}.csv"

            
            
            logger.info(f"Data saved  {out_path}")
        
            
            return transformed_df
        
        except ClientError as e:
            raise
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise

    def validate_schema(self):
        try:
            logger.info("")
            logger.info("STEP 0 - SCHEMA CHECK")
            logger.info("-" * 50)

            missing_cols = set(self.config.expected_columns) - set(self.df.columns)

            if missing_cols:
                logger.error(f"FAILED - Missing columns: {missing_cols}")
                return f"Missing columns: {missing_cols}"

            if "timestamp" in self.df.columns:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)

            if "bot_score" in self.df.columns:
                self.df["bot_score"] = self.df["bot_score"].astype(float)

            logger.info("PASSED - Step 0 Schema Check")
            return None

        except Exception as e:
            logger.error(f"Schema validation aborted: {str(e)}")
            return f"Schema validation aborted: {str(e)}"


    def validate_null_values(self):
        try:
            logger.info("")
            logger.info("STEP 1 - NULL VALUE CHECK")
            logger.info("-" * 50)

            if self.df.isnull().values.any():
                null_counts = self.df.isnull().sum()
                logger.error(f"FAILED - NULL VALUES FOUND:\n{null_counts[null_counts > 0]}")
                return "Dataset contains null values. Validation failed."

            logger.info("PASSED - Step 1 Null Value Check")
            return None

        except Exception as e:
            logger.error(f"Null validation aborted: {str(e)}")
            return f"Null validation aborted: {str(e)}"


    def validate_volume(self):
        try:
            logger.info("")
            logger.info("STEP 2 - VOLUME CHECK")
            logger.info("-" * 50)

            min_rows  = self.config.volume["min_rows"]
            max_rows  = self.config.volume["max_rows"]
            row_count = len(self.df)

            logger.info(f"Rows found: {row_count}")

            if row_count < min_rows:
                logger.error(f"FAILED - VOLUME TOO LOW: Expected at least {min_rows} rows, but found {row_count}.")
                return f"Dataset too small: {row_count} rows."

            if row_count > max_rows:
                logger.warning(f"WARNING - VOLUME HIGH: Expected max {max_rows} rows, but found {row_count}.")

            logger.info(f"PASSED - Step 2 Volume Check ({row_count} rows)")
            return None

        except Exception as e:
            logger.error(f"Volume validation aborted: {str(e)}")
            return f"Volume validation aborted: {str(e)}"


    def validate_temporal_integrity(self):
        try:
            logger.info("")
            logger.info("STEP 3 - TEMPORAL INTEGRITY CHECK")
            logger.info("-" * 50)

            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

            if self.config.temporal["require_sorted"]:
                if not self.df['timestamp'].is_monotonic_increasing:
                    logger.error("FAILED - Timestamp column is not sorted.")
                    return "Data is not sorted chronologically."

            diffs             = self.df['timestamp'].diff().dropna()
            expected_interval = pd.Timedelta(minutes=self.config.temporal["interval_minutes"])
            gaps              = diffs[diffs != expected_interval]

            if len(gaps) > self.config.temporal["max_gap_count"]:
                logger.error(f"FAILED - Found {len(gaps)} gaps that do not match {expected_interval} interval.")
                return f"Temporal gaps detected: {len(gaps)} violations."

            logger.info("PASSED - Step 3 Temporal Integrity Check")
            return None

        except Exception as e:
            logger.error(f"Temporal validation aborted: {str(e)}")
            return f"Temporal validation aborted: {str(e)}"


    def validate_statistical_drift(self):
        try:
            logger.info("")
            logger.info("STEP 4 - STATISTICAL DRIFT CHECK")
            logger.info("-" * 50)

            stats        = self.config.statistical
            current_mean = self.df['bot_score'].mean()
            current_std  = self.df['bot_score'].std()

            logger.info(f"Current Mean : {current_mean:.4f}  |  Baseline : {stats['baseline_mean']}")
            logger.info(f"Current Std  : {current_std:.4f}  |  Baseline : {stats['baseline_std']}")

            if current_std < stats["min_std"]:
                logger.error(f"FAILED - Variance too low ({current_std:.6f}). Data might be corrupted.")
                return "Suspiciously identical values detected in bot_score."

            drift_threshold = stats["drift_factor"] * stats["baseline_std"]
            if abs(current_mean - stats["baseline_mean"]) > drift_threshold:
                logger.error(f"FAILED - Mean {current_mean:.4f} drifted from baseline {stats['baseline_mean']:.4f}")
                return "Data drift detected: statistical distribution has changed."

            logger.info(f"PASSED - Step 4 Statistical Drift Check")
            return None

        except Exception as e:
            logger.error(f"Statistical validation aborted: {str(e)}")
            return f"Statistical validation aborted: {str(e)}"


    def validate_range_check(self):
        try:
            logger.info("")
            logger.info("STEP 5 - RANGE CHECK")
            logger.info("-" * 50)

            bot_score_range = self.config.range_checks["bot_score"]

            violations = self.df[
                (self.df['bot_score'] < bot_score_range["min"]) |
                (self.df['bot_score'] > bot_score_range["max"])
            ]

            if not violations.empty:
                logger.error(f"FAILED - {len(violations)} records outside [{bot_score_range['min']}, {bot_score_range['max']}].")
                logger.error(f"Sample invalid values: {violations['bot_score'].head().tolist()}")
                return f"Range validation failed: {len(violations)} out-of-bounds records."

            logger.info("PASSED - Step 5 Range Check")
            return None

        except Exception as e:
            logger.error(f"Range validation aborted: {str(e)}")
            return f"Range validation aborted: {str(e)}"
        
    def _save_data(self, quarantine: bool):
        try:
            if quarantine:
                folder = Path(self.config.quarantine_data)
                label  = "quarantine"
            else:
                folder = folder = Path(self.config.validated_data) / f"lambda_run__{self.pipeline_run_id}"
                label  = "validated"

            folder.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            out_path  = folder / f"{label}_{timestamp}.csv"

            self.df.to_csv(out_path, index=False)

            logger.info(f"Data saved  {out_path}")

        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise

    def run(self):
        try:
            logger.info("=" * 50)
            logger.info("DATA VALIDATION PIPELINE STARTED")
            logger.info("=" * 50)

            results = {
                "schema"      : self.validate_schema(),
                "missing"     : self.validate_null_values(),
                "volume"      : self.validate_volume(),
                "temporal"    : self.validate_temporal_integrity(),
                "statistical" : self.validate_statistical_drift(),
                "range"       : self.validate_range_check(),
            }

            failed_checks = [
                check for check, error in results.items()
                if error is not None
            ]

            if failed_checks:
                logger.info("-" * 50)
                logger.info("VALIDATION ERRORS SUMMARY:")
                for check in failed_checks:
                    logger.error(f"  [{check.upper()}] -> {results[check]}")
                logger.info("-" * 50)

            hard_failures = [c for c in failed_checks if c in self.config.hard_fail_checks]
            soft_failures = [c for c in failed_checks if c in self.config.soft_fail_checks]

            if hard_failures:
                logger.info("")
                logger.error(f"HARD FAIL - checks failed: {hard_failures}")
                logger.error("Data is being sent to quarantine.")
                self._save_data(quarantine=True)
                logger.info("=" * 50)
                logger.info("DATA VALIDATION COMPLETE - QUARANTINED")
                logger.info("=" * 50)
                return False

            if soft_failures:
                logger.info("")
                logger.warning(f"SOFT FAIL - warnings on: {soft_failures}")
                logger.warning("Data passes to validated folder with warnings.")
                self._save_data(quarantine=False)

                output_key = f"data_validation/run__{self.pipeline_run_id}/validation_df.csv"
                
                manifest = save_manifest(output_key=output_key,pipline_id=self.pipeline_run_id)

                save_file_s3(df=self.df ,output_key=output_key ,BUCKET_NAME=self.BUCKET_NAME,s3_client=self.s3)

                self_s3_mainfest(manifest=manifest ,output_key=output_key,BUCKET_NAME=self.BUCKET_NAME,s3_client=self.s3)


                self_local_save_mainfest(manifest=manifest,local_dir=self.config.validated_data,pipeline_run_id=self.pipeline_run_id)

                
                logger.info("=" * 50)
                logger.info("DATA VALIDATION COMPLETE - PASSED WITH WARNINGS")
                logger.info("=" * 50)
                return True

            logger.info("")
            logger.info("ALL CHECKS PASSED")
            self._save_data(quarantine=False)

            
            logger.info("=" * 50)
            logger.info("DATA VALIDATION COMPLETE - PASSED")
            logger.info("=" * 50)
            return True
            

        except Exception as e:
            logger.error(f"Validation pipeline aborted: {str(e)}")
            raise



        


    
        
        
        

    

    



    
        
        
        
