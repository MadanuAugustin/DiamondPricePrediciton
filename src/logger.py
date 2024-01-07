import logging
import os
from datetime import datetime

my_log_files = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
my_logs_path = os.path.join(os.getcwd(), "my_log_files", my_log_files)
os.makedirs(my_logs_path, exist_ok= True)


LOGS_FILE_PATH = os.path.join(my_logs_path, my_log_files)


logging.basicConfig(
    filename = LOGS_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)

logging.info('the loggers file has been executed successfully...!')