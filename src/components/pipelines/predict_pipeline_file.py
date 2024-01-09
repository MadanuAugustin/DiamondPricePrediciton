import sys
import pandas as pd
from src.exception_file import my_exception
from src.utils_file import load_object


class PredictPipeline:
    def __init__(self):
        pass

    
    def predict(self, features):
        try:
            model_path = 'pickle_files\\model.pkl'
            preprocessor_path = 'pickle_files\\preprocessor_obj.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as error:
            raise my_exception(error, sys)
        

class CustomData:
    def __init__(self,
                 color,
                 clarity,
                 depth,
                 carat,
                 table,
                 cut,
                 x,
                 y,
                 z):
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.carat = carat
        self.table = table
        self.cut = cut
        self.x = x
        self.y = y
        self.z = z


    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "color" : [self.color],
                "clarity" : [self.clarity],
                "depth" : [self.depth],
                "carat" : [self.carat],
                "table" : [self.table],
                "cut" : [self.cut],
                "x" : [self.x],
                "y" : [self.y],
                "z" : [self.z]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as error:
            raise my_exception(error, sys)