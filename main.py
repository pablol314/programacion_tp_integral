#main.py

import pandas as pd
import numpy as np
from collections import Counter
from core.logging_config import logger
from preprocessing_data import preprocessing_data

logging = logger

# Cargar y preprocesar datos
df = preprocessing_data(path_data='data/data.csv')

# Definir umbral de pico de audiencia por canal
df['umbral_pico'] = df.groupby('channel_name')['concurrent_view_count'].transform(lambda x: x.quantile(0.75))
df['es_pico'] = df['concurrent_view_count'] >= df['umbral_pico']

# Agrupar canales poco frecuentes
top_canales = df['channel_name'].value_counts().nlargest(10).index
if 'otros' not in df['channel_name'].cat.categories:
    df['channel_name'] = df['channel_name'].cat.add_categories(['otros'])
df['channel_name_mod'] = df['channel_name'].where(df['channel_name'].isin(top_canales), 'otros')

# Balancear clases con submuestreo
pico_df = df[df['es_pico']]
no_pico_df = df[~df['es_pico']].sample(n=len(pico_df), random_state=40)
df_balanceado = pd.concat([pico_df, no_pico_df])

# Features y etiquetas
X = df_balanceado[['hour', 'weekday', 'channel_type', 'channel_name_mod']]
y = df_balanceado['es_pico']
X = pd.get_dummies(X)

# Eliminar columnas de baja varianza
X = X.loc[:, X.var() > 0.01]

# Árbol de decisión
def entropia(y):
    total = len(y)
    if total == 0:
        return 0
    conteo = Counter(y)
    return -sum((n / total) * np.log2(n / total) for n in conteo.values())

def ganancia_info(X_col, y, umbral):
    izquierda = X_col <= umbral
    derecha = ~izquierda
    n = len(y)
    if izquierda.sum() == 0 or derecha.sum() == 0:
        return 0
    ent_total = entropia(y)
    ent_izq = entropia(y[izquierda])
    ent_der = entropia(y[derecha])
    ent_ponderada = (izquierda.sum() / n) * ent_izq + (derecha.sum() / n) * ent_der
    return ent_total - ent_ponderada

def mejor_division(X, y):
    mejor_ganancia = -1
    mejor_atributo = None
    mejor_umbral = None
    for atributo in X.columns:
        for umbral in X[atributo].unique():
            ganancia = ganancia_info(X[atributo], y, umbral)
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo
                mejor_umbral = umbral
    return mejor_atributo, mejor_umbral

class Nodo:
    def __init__(self, atributo=None, umbral=None, izquierda=None, derecha=None, valor=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izquierda = izquierda
        self.derecha = derecha
        self.valor = valor

def construir_arbol(X, y, profundidad=0, profundidad_max=12):
    if len(y) == 0:
        return Nodo(valor=False)  # valor por defecto si no hay datos

    if len(set(y)) == 1 or profundidad >= profundidad_max:
        valor = y.mode().iloc[0] if not y.mode().empty else False
        return Nodo(valor=valor)

    atributo, umbral = mejor_division(X, y)
    if atributo is None:
        valor = y.mode().iloc[0] if not y.mode().empty else False
        return Nodo(valor=valor)

    izquierda = X[atributo] <= umbral
    derecha = ~izquierda

    nodo_izq = construir_arbol(X[izquierda], y[izquierda], profundidad + 1, profundidad_max)
    nodo_der = construir_arbol(X[derecha], y[derecha], profundidad + 1, profundidad_max)
    return Nodo(atributo, umbral, nodo_izq, nodo_der)


def predecir(nodo, fila):
    if nodo.valor is not None:
        return nodo.valor
    if fila[nodo.atributo] <= nodo.umbral:
        return predecir(nodo.izquierda, fila)
    else:
        return predecir(nodo.derecha, fila)

def imprimir_arbol(nodo, profundidad=0):
    sangria = "  " * profundidad
    if nodo.valor is not None:
        logging.info(f"{sangria}-> {'PICO' if nodo.valor else 'NO PICO'}")
        return
    logging.info(f"{sangria}if {nodo.atributo} <= {nodo.umbral}:")
    imprimir_arbol(nodo.izquierda, profundidad + 1)
    logging.info(f"{sangria}else:")
    imprimir_arbol(nodo.derecha, profundidad + 1)

# Entrenamiento
logging.info("Entrenando árbol de decisión...")
arbol = construir_arbol(X, y, profundidad_max=6)

# Predicción
logging.info("Evaluando árbol...")
y_pred = X.apply(lambda fila: predecir(arbol, fila), axis=1)

# Métricas manuales
tp = sum((y == True) & (y_pred == True))
tn = sum((y == False) & (y_pred == False))
fp = sum((y == False) & (y_pred == True))
fn = sum((y == True) & (y_pred == False))

precision = (tp + tn) / len(y)
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

logging.info(f"Precisión global: {precision:.2%}")
logging.info(f"Precisión (Pico): {precision_pos:.2%}")
logging.info(f"Recall (Pico): {recall_pos:.2%}")
logging.info(f"F1-score: {f1:.2%}")

# Mostrar árbol entrenado
logging.info("Estructura del árbol:")
imprimir_arbol(arbol)

# Guardar predicciones
df_balanceado['prediccion_pico'] = y_pred

# Gráficos
import graficos
graficos.grafico_comparacion_pico_real_vs_predicho(df_balanceado)

