from core.logging_config import logger
from preprocessing.cargar_y_preprocesar import preparar_datos
from modelo_arbol.arbol_decision import construir_arbol, predecir, imprimir_arbol
from modelo_arbol.metricas import calcular_metricas
import graficos

logging = logger

# Datos
X, y, df_balanceado = preparar_datos('data/processed_data.csv')

# Entrenamiento
logging.info("Entrenando árbol de decisión...")
arbol = construir_arbol(X, y, profundidad_max=6)

# Predicción
logging.info("Evaluando árbol...")
y_pred = X.apply(lambda fila: predecir(arbol, fila), axis=1)

# Métricas
calcular_metricas(y, y_pred)

# Mostrar árbol
logging.info("Estructura del árbol:")
imprimir_arbol(arbol)

# Guardar resultados
df_balanceado['prediccion_pico'] = y_pred

# Gráficos
graficos.grafico_comparacion_pico_real_vs_predicho(df_balanceado)
