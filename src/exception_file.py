import sys
import logging
from src.logger import logging

# Here we are creating our own custom exception class which is my_exception and inheriting the properties of built-in Exception class
class my_exception(Exception):
    def __init__(self, my_error_msg, system_error:sys):
        self.my_error_msg = error_function(my_error_msg = my_error_msg, system_error = system_error)
        # super().__init__(my_error_msg)
        super(my_exception, self).__init__(my_error_msg)    ## we are overiding the constructor of the base Exception class to accept our own arguments


    
    def __str__(self):
        logging.info('the exception file has been executed successfully...!')
        return self.my_error_msg
    


# the below function return the error msg
    
def error_function(my_error_msg, system_error:sys):
    _,_,exc_tb = system_error.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    my_error_msg = "The error has been raised in python script name {} line number {} and error message is : {}".format(filename, line_no, my_error_msg)

    return my_error_msg


