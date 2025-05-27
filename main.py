import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('data/data.csv')
# Convertir datetime_now a formato datetime
df['datetime_now'] = pd.to_datetime(df['datetime_now'], errors='coerce')

# Ver cantidad de valores nulos por columna
print(df.isnull().sum())

# Opcional: eliminar filas con datos nulos en columnas importantes
df.dropna(subset=['datetime_now', 'avg_concurrent_views', 'channel_name', 'channel_type'], inplace=True)

# Convertir columnas numéricas si están en formato incorrecto
df['avg_concurrent_views'] = pd.to_numeric(df['avg_concurrent_views'], errors='coerce')

# Convertir 'channel_type' y 'channel_name' a categoría
df['channel_type'] = df['channel_type'].astype('category')
df['channel_name'] = df['channel_name'].astype('category')

# Separar fecha y hora
df['date'] = df['datetime_now'].dt.date
df['hour'] = df['datetime_now'].dt.hour
df['weekday'] = df['datetime_now'].dt.day_name()

# Filtrar columnas innecesarias
columns_of_interest = [
    'datetime_now',
    'avg_concurrent_views',
    'channel_name',
    'channel_type',
    'date',
    'hour',
    'weekday'
]
df = df[columns_of_interest]

# Guardar una versión preprocesada del dataset
df.to_csv('data/processed_data.csv', index=False)

print(df.head())
print(df.describe())
print(df.dtypes)
