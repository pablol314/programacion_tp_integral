import pandas as pd
from core.logging_config import logger
from preprocessing_data import preprocessing_data
from sklearn.model_selection import train_test_split

logging = logger

# Cargar y preprocesar los datos
df = preprocessing_data(path_data='data/data.csv')

# Filtrar los datos: miércoles (día 2), entre 20:00 y 22:00
df_filtrado = df[
    (df['weekday'] == 2) &  # Miércoles
    (df['hour'] >= 20) &
    (df['hour'] < 22)
].copy()

print(f"Filas en el rango miércoles 20 a 22 hs: {df_filtrado.shape[0]}")

# Verificar si hay datos suficientes para procesar
if df_filtrado.empty:
    print("No hay datos disponibles para la franja horaria seleccionada.")
else:
    # Mostrar ranking de canales por visitas
    visitas_por_canal = df_filtrado.groupby('channel_name')['concurrent_view_count'].sum().sort_values(ascending=False)

    print("\n🎯 Ranking de canales por cantidad de espectadores (miércoles 20 a 22 hs):\n")
    for canal, visitas in visitas_por_canal.items():
        print(f"{canal}: {visitas} espectadores")

    # Definir la clase objetivo en base a la cantidad de espectadores
    df_filtrado['success'] = (df_filtrado['concurrent_view_count'] > df_filtrado['concurrent_view_count'].median()).astype(int)

    # Preparar variables
    X = df_filtrado.drop(columns=['success'])
    y = df_filtrado['success']

    # Dividir el conjunto de datos
    train_df, test_df = train_test_split(df_filtrado, test_size=0.2, stratify=df_filtrado['success'], random_state=42)

    print("Conjunto de entrenamiento:", train_df.shape)
    print("Conjunto de validación:", test_df.shape)

    # Estructura del árbol
    class Nodo:
        def __init__(self, atributo=None, umbral=None, izquierda=None, derecha=None, clase=None):
            self.atributo = atributo
            self.umbral = umbral
            self.izquierda = izquierda
            self.derecha = derecha
            self.clase = clase

    def entropia(y):
        from collections import Counter
        import numpy as np

        total = len(y)
        if total == 0:
            return 0
        counts = Counter(y)
        return -sum((count / total) * np.log2(count / total) for count in counts.values() if count > 0)

    def ganancia_info(X_col, y, umbral):
        parent_entropy = entropia(y)
        left_indices = X_col <= umbral
        right_indices = X_col > umbral
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])
        weighted_entropy = (n_left / n) * entropia(y[left_indices]) + (n_right / n) * entropia(y[right_indices])
        return parent_entropy - weighted_entropy

    def mejor_division(X, y):
        mejor_ganancia = 0
        mejor_atributo = None
        mejor_umbral = None

        for atributo in X.select_dtypes(include=['int', 'float']).columns:
            umbrales = X[atributo].unique()
            for umbral in umbrales:
                ganancia = ganancia_info(X[atributo], y, umbral)
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_atributo = atributo
                    mejor_umbral = umbral

        return mejor_atributo, mejor_umbral

    def construir_arbol(X, y):
        if len(set(y)) == 1:
            return Nodo(clase=y.iloc[0])

        if X.empty:
            return Nodo(clase=y.mode()[0])

        atributo, umbral = mejor_division(X, y)
        if atributo is None:
            return Nodo(clase=y.mode()[0])

        left_indices = X[atributo] <= umbral
        right_indices = X[atributo] > umbral

        izquierda = construir_arbol(X[left_indices], y[left_indices])
        derecha = construir_arbol(X[right_indices], y[right_indices])

        return Nodo(atributo=atributo, umbral=umbral, izquierda=izquierda, derecha=derecha)

    # Construcción del árbol
    arbol = construir_arbol(X, y)


