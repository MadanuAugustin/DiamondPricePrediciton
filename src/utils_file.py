import os
import sys
import numpy as np
import pandas as pd
import dill


from src.exception_file import my_exception
from src.logger import logging


def save_preprocessor_obj(file_path, obj):
    try:
        os.makedirs("pickle_files", exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info('successfully saved preprocessor obj as a pickle file...!')

    except Exception as error:
        raise my_exception(error, sys)