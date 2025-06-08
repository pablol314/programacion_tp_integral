import numpy as np
from collections import Counter

class Nodo:
    def __init__(self, atributo=None, umbral=None, izquierda=None, derecha=None, valor=None):
        self.atributo = atributo
        self.umbral = umbral
        self.izquierda = izquierda
        self.derecha = derecha
        self.valor = valor

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

def construir_arbol(X, y, profundidad=0, profundidad_max=12):
    if len(y) == 0:
        return Nodo(valor=False)
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
    from core.logging_config import logger
    sangria = "  " * profundidad
    if nodo.valor is not None:
        logger.info(f"{sangria}-> {'PICO' if nodo.valor else 'NO PICO'}")
        return
    logger.info(f"{sangria}if {nodo.atributo} <= {nodo.umbral}:")
    imprimir_arbol(nodo.izquierda, profundidad + 1)
    logger.info(f"{sangria}else:")
    imprimir_arbol(nodo.derecha, profundidad + 1)
