import os
import sys
from pathlib import Path 
import logging 

project_name = 'predictor-bot-score'
files = [

    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/constants/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/logging/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/config/__init__.py',
    f"src/{project_name}/config/configuration.py"
    'main.py'
]

for i in files:
    path = Path(i)
    file_dir , file_name = os.path.split(path)
    logging.info(f"Creating directory:{file_dir} for the file {file_name}")


    if file_dir != " ":
        os.makedirs(file_dir,exist_ok=True)
        with open(path,'w') as f:
            pass 


