import os
from pathlib import Path
import logging

project_name = "predictor_bot_score"

files = [
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/logger/__init__.py",   # renamed
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    "main.py"
]

for filepath in files:

    filepath = Path(filepath)

    file_dir, file_name = os.path.split(filepath)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)

    logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    with open(filepath, "w") as f:
        pass