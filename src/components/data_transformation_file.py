import sys
import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception_file import my_exception
from src.utils_file import save_preprocessor_obj

class Transformation:
    def __init__(self):
         pass


    def transformer_obj(self):

        try:
            numerical_columns = ['carat', 'depth', 'table', 'x','y', 'z']
            categorical_columns = ['cut','color','clarity']

            logging.info('creating numerical and categorical pipelines...!')
        
            numeric_pipeline = Pipeline(
                steps =[
                    ('imputer', SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
            )

            categorical_pipeline = Pipeline(
                steps =[
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("ordinaly_encoding", OrdinalEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info('successfully created numerical and categorical pipelines...!')

            logging.info('creating preprocessor...!')

            preprocessor = ColumnTransformer(
                [
                ("numericPipeline", numeric_pipeline, numerical_columns),
                ('categoricPipeline', categorical_pipeline, categorical_columns)
                ]
            )
            logging.info('successfully created preprocessor obj...!')

            return preprocessor
        
    
        except Exception as error:
            raise my_exception(error, sys)
        

    def transforming_data(self, train_set, test_set):
            try:
                train_df = train_set
                test_df = test_set

                logging.info('successfully read train_df and test_df...!')

                target_column = 'price'

                input_feature_train_df = train_df.drop(columns = [target_column], axis = 1)
                target_feature_train_df = train_df[target_column]

                input_feature_test_df = test_df.drop(columns = [target_column], axis = 1)
                target_feature_test_df = test_df[target_column]

                preprocessor_obj = self.transformer_obj()

                logging.info('applying preprocessor_obj on the train and test data...!')

                input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                train_arr_df = pd.DataFrame(train_arr)
                test_arr_df = pd.DataFrame(test_arr)
                
                train_arr_df.to_csv('artifact_files\\transformed_train_data.csv', header=True, index = False)
                test_arr_df.to_csv('artifact_files\\transformed_test_data.csv', header=True, index = False)


                logging.info('successfully transformed train and test data using preprocessor_obj...!')

                save_preprocessor_obj(
                    file_path = "pickle_files\\preprocessor_obj.pkl",
                    obj = preprocessor_obj
                )

                return(
                    train_arr, test_arr
                )

            except Exception as error:
                raise my_exception(error, sys)    