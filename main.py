import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('data/data.csv')
print(df.head)
print(df.describe())
print(df.dtypes)