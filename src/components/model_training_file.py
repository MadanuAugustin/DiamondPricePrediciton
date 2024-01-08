import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from src.exception_file import my_exception
from src.logger import logging

from src.utils_file import evaluate_models, save_preprocessor_obj

class Model_trainer:
    def __init__(self):
        pass

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('splitting the data into x_train, y_train, x_test, y_test...!')
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:, -1:], test_arr[:, :-1], test_arr[:, -1:])

            models = {
                "LinearRegression" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "SVR" : SVR(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor()
            }

            hyper_tunning = {
                "LinearRegression":{},
                "Ridge" : {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]},
                "Lasso" : {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]},
                "SVR" : {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
                "DecisionTreeRegressor":{'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                "KNeighborsRegressor":{"n_neighbors":5},
                "AdaBoostRegressor":{"estimator":None, "n_estimators":50, "learning_rate":1.0},
                "GradientBoostingRegressor":{ "learning_rate":0.1, "n_estimators":100},
                "RandomForestRegressor":{'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
                "XGBRegressor":{"n_estimators" :[100, 500, 900, 1100, 1500],"max_depth" :[2, 3, 5, 10, 15],"learning_rate":[0.05,0.1,0.15,0.20],"min_child_weight":[1,2,3,4]}
            }


            model_report : dict=evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = hyper_tunning)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise my_exception("No best model found...!")
            
            logging.info('Best model found...!')
            logging.info('saving the model as a pickle file...!')
            save_preprocessor_obj(
                file_path="pickle_files\\model.pkl",
                obj = best_model
            )

            return(
                best_model , best_model_score
            )

        except Exception as error:
            raise my_exception(error, sys)

