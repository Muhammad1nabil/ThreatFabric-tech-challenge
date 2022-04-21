from msilib.schema import Error
from multiprocessing.connection import wait
import pathlib
import os
from time import sleep

cwd = pathlib.Path.cwd()

if (not (cwd / 'venv').exists()):
    if ((cwd / 'requirements.txt').exists()) and ((cwd / 'xgboost_wheel' / 'xgboost-1.5.1-cp38-cp38-win32.whl')):
        os.system('create_venv.bat && python flask_app.py')    
    else:
        raise FileNotFoundError('requirements.txt/xgboost wheel is not found!')

