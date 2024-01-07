import os
import sys
import pandas as pd
import logging

from src.exception_file import my_exception
from src.logger import logging

from sklearn.model_selection import train_test_split

from src.components.data_transformation_file import Transformation


class DataLoading:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        logging.info('starting the data ingestion process...!')

        try:
            mydf = pd.read_csv('JupyterNotebook\\Data\\diamonds.csv')
            logging.info('read the dataset successfully and converted to dataframe...!')

            os.makedirs('artifact_files', exist_ok=True)

            mydf.to_csv('artifact_files\\raw_data.csv')

            logging.info('starting the train_test_split process...!')
            train_set, test_set = train_test_split(mydf, test_size=0.2, random_state=42)

            train_set.to_csv('artifact_files\\train_set.csv')

            test_set.to_csv('artifact_files\\test_set.csv')

            logging.info('train_test_split completed and data ingestion process completed...!')

            return(
                train_set, test_set
            )
        except Exception as error:
            raise my_exception(error, sys)
            

if __name__ == '__main__':
    myobj = DataLoading()
    train_set, test_set = myobj.initiate_data_ingestion()

    data_transformation = Transformation()
    train_arr, test_arr = data_transformation.transforming_data(train_set, test_set)