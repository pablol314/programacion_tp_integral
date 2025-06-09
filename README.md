# 📊 Detección de Picos de Audiencia mediante Árboles de Decisión en Python

**Trabajo Integrador - Programación I**  
Tecnicatura Universitaria en Programación  
Universidad Tecnológica Nacional (UTN)  
Comisión 16

## 👥 Integrantes
- Pablo León  
- Andrés Piuzzi  

## 🎯 Objetivo

Este proyecto tiene como objetivo desarrollar un modelo predictivo utilizando un **árbol de decisión implementado desde cero en Python** para detectar **momentos de alta audiencia (picos)** en transmisiones en vivo en plataformas de streaming, a partir del análisis de datos temporales y categóricos.

---

## 📁 Estructura del Proyecto

- `informe.pdf`: Informe completo del trabajo práctico.
- `arbol_decision.py`: Implementación del árbol de decisión desde cero.
- `preprocesamiento.py`: Código para cargar, limpiar y transformar los datos.
- `dataset/`: Carpeta con datos anonimizados usados en el análisis.
- `notebooks/`: Análisis exploratorio, visualizaciones y pruebas.
- `README.md`: Este archivo.

---

## 📊 Metodología

1. **Recolección de datos:** scraping de plataformas de streaming (cada 5 minutos).
2. **Preprocesamiento:**
   - Normalización horaria
   - Agrupamiento de canales poco frecuentes
   - Generación de variable objetivo (`pico` si supera el percentil 75)
   - Codificación de variables categóricas (`dummies`)
   - Balanceo de clases con submuestreo
3. **Construcción del árbol:**
   - Cálculo de entropía
   - Ganancia de información
   - División recursiva hasta nodo puro o profundidad límite
4. **Evaluación:**  
   - Precisión global: **74%**
   - Análisis de reglas generadas
   - Validación de tendencias horarias

---

## 📌 Resultados Destacados

- Reglas como:  
  > *"Si el canal es de tipo TV y transmite entre las 21:00 y 23:00, hay alta probabilidad de pico de audiencia"*

- Las franjas horarias con mayor concurrencia detectadas fueron:
  - **10:00–12:00**
  - **21:00–23:00**

---

## 📺 Video del proyecto

▶️ [Ver en YouTube](https://www.youtube.com/watch?v=5XGH7UwnH4w)

---

## 📄 Informe final

📝 [Informe Google Docs](https://docs.google.com/document/d/1jPzpIeFYGhfunefu3rNap2XHFQfVTWLzE31EAIKllk0/edit?usp=sharing)

---

## 📚 Bibliografía

- Documentación oficial de Python: https://docs.python.org/3  
- Pandas Docs: https://pandas.pydata.org  
- Numpy Docs: https://numpy.org/doc  
- Visualgo: https://visualgo.net  
- Material de cátedra Programación I – UTN  

---


