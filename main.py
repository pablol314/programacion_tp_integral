import pandas as pd
from core.logging_config import logger
from preprocessing_data import preprocessing_data

logging = logger

df = preprocessing_data(path_data='data/data.csv')

hola = "hola2"

from sklearn.model_selection import train_test_split

# Definir la clase objetivo
df['success'] = (df['avg_concurrent_views'] > df['avg_concurrent_views'].median()).astype(int)

# Dividir el conjunto de datos
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['success'], random_state=42)

print("Conjunto de entrenamiento:", train_df.shape)
print("Conjunto de validación:", test_df.shape)
