#procesamiento_raw.py

import pandas as pd
from core.logging_config import logger

logging = logger

def preprocessing_data(path_data='data/data.csv'):
    logging.info("Cargando el archivo CSV...")
    df = pd.read_csv(path_data)
    logging.info(f"Archivo cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Conversión de datetime
    logging.info("Convirtiendo 'datetime_now' a formato datetime...")
    df['datetime_now'] = pd.to_datetime(df['datetime_now'], errors='coerce')

    # Revisión de valores nulos
    null_counts = df.isnull().sum()
    logging.info(f"Valores nulos por columna:\n{null_counts}")

    # Eliminación de filas con valores nulos en columnas clave
    before_drop = df.shape[0]
    df.dropna(subset=['datetime_now', 'concurrent_view_count', 'channel_name', 'channel_type'], inplace=True)
    after_drop = df.shape[0]
    logging.info(f"Filas eliminadas por valores nulos: {before_drop - after_drop}")

    # Conversión de tipo numérico
    logging.info("Convirtiendo 'concurrent_view_count' a tipo numérico...")
    df['concurrent_view_count'] = pd.to_numeric(df['concurrent_view_count'], errors='coerce')

    # Conversión de tipo categórico
    logging.info("Convirtiendo 'channel_type' y 'channel_name' a tipo categoría...")
    df['channel_type'] = df['channel_type'].astype('category')
    df['channel_name'] = df['channel_name'].astype('category')

    # Normalización temporal
    logging.info("Normalizando la columna 'hour'...")
    df['hour'] = df['datetime_now'].dt.floor('h')

    # Filtrado de franjas representativas por canal
    logging.info("Filtrando franjas representativas por canal...")
    df['hour_count'] = df.groupby(['channel_name', 'hour'])['datetime_now'].transform('count')
    avg_hour_count = df.groupby('channel_name')['hour_count'].transform('mean')
    df = df[df['hour_count'] >= avg_hour_count]
    logging.info("Creando columnas de fecha, hora y día de la semana...")
    df['date'] = df['datetime_now'].dt.date
    df['hour'] = df['datetime_now'].dt.hour
    df['weekday'] = df['datetime_now'].dt.weekday

    # Filtrado de columnas relevantes
    columns_of_interest = [
        'datetime_now',
        'concurrent_view_count',
        'channel_name',
        'channel_type',
        'date',
        'hour',
        'weekday'
    ]
    df = df[columns_of_interest]
    logging.info(f"Filtrado de columnas completado. Columnas actuales: {df.columns.tolist()}")


    # Mostrar resumen final
    logging.info(f"Primeras filas del dataframe:\n{df.head()}")
    logging.info(f"Resumen estadístico:\n{df.describe(include='all')}")
    logging.info(f"Tipos de datos:\n{df.dtypes}")

    return df
