from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Data_injestion_config:

    bucket_name : str 
    file_name  : str
    raw_data : Path
    ingested_data : Path

