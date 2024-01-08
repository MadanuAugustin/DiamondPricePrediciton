import os
import sys
import numpy as np
import pandas as pd
import dill


from src.exception_file import my_exception
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_preprocessor_obj(file_path, obj):
    try:
        os.makedirs("pickle_files", exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info('successfully saved preprocessor obj as a pickle file...!')

    except Exception as error:
        raise my_exception(error, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            test_model_score = r2_score(y_test, predicted)
            report[list(models.keys())[i]] = test_model_score

            return report
    except Exception as error:
        raise my_exception(error, sys)