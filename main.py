import pandas as pd
from core.logging_config import logger
from preprocessing_data import preprocessing_data

logging = logger

df = preprocessing_data(path_data='data/data.csv')

hola = "hola2"

print(df)