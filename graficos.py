#graficos.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grafico_comparacion_pico_real_vs_predicho(df):
    resumen = df.groupby('hour').agg(
        real_pico=('es_pico', 'mean'),
        pred_pico=('prediccion_pico', 'mean')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(resumen['hour'], resumen['real_pico'], label='Pico Real')
    plt.plot(resumen['hour'], resumen['pred_pico'], label='Pico Predicho', linestyle='--')
    plt.xlabel('Hora del día')
    plt.ylabel('Proporción de picos')
    plt.title('Comparación: Picos Reales vs Predichos por Hora')
    plt.legend()
    plt.show()

