from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Data_injestion_config:

    bucket_name : str 
    file_name  : str
    raw_data : Path
    ingested_data : Path

@dataclass(frozen=True)
class DataValidationConfig:

    # --- paths ---
    raw_data_path   : Path
    validated_data  : Path
    quarantine_data : Path
    log_path        : Path

    # --- step 1 : schema ---
    expected_columns : list[str]
    expected_dtypes  : dict[str, str]

    # --- step 2 : data quality ---
    missing_thresholds      : dict[str, float]
    allow_duplicate_timestamps : bool
    volume                  : dict[str, int]

    # --- step 3 : temporal ---
    temporal : dict[str, object]

    # --- step 4 : statistical ---
    statistical : dict[str, float]

    # --- step 5 : feature validation ---
    range_checks  : dict[str, dict]
    outlier_floor : float
    max_step_drop : float

    # --- step 6 : business logic ---
    business_logic : dict[str, object]

    # --- outcome routing ---
    hard_fail_checks : list[str]
    soft_fail_checks : list[str]

