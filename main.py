import pandas as pd
from core.logging_config import logger
from preprocessing_data import preprocessing_data

logging = logger

df = preprocessing_data(path_data='data/data.csv')

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

    for atributo in X.columns:
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

# Entrenar el árbol de decisión
X = df.drop(columns=['success'])
y = df['success']
arbol = construir_arbol(X, y)


# Definir la clase objetivo
df['success'] = (df['avg_concurrent_views'] > df['avg_concurrent_views'].median()).astype(int)

# Dividir el conjunto de datos
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['success'], random_state=42)

print("Conjunto de entrenamiento:", train_df.shape)
print("Conjunto de validación:", test_df.shape)
