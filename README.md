#  CLASIFICACIÓN AVANZADA DE CÁNCER DE MAMA (Breast Cancer Wisconsin) 

Este proyecto implementa y compara múltiples modelos de Machine Learning para clasificar tumores mamarios como **Benignos** (Clase 1) o **Malignos** (Clase 0), utilizando el popular conjunto de datos Breast Cancer Wisconsin (Diagnóstico) de scikit-learn.

El objetivo principal es identificar el modelo con la **mayor capacidad predictiva** (medida principalmente por el área bajo la curva ROC, **AUC-ROC**) para asistir en el diagnóstico temprano con alta fiabilidad.

## Configuración y Librerías

El análisis fue ejecutado utilizando las siguientes bibliotecas clave de Python:

* **Scikit-learn (sklearn):** Para la carga del dataset, preprocesamiento, implementación de modelos y métricas.
* **NumPy:** Para operaciones numéricas eficientes.
* **Pandas:** Para manipulación y exploración de datos.
* **Matplotlib y Seaborn:** Para la generación de gráficos de visualización de resultados.

##  Dataset y Preprocesamiento

### Resumen del Dataset

| Característica | Valor |
| :--- | :--- |
| **Total de Muestras** | 569 |
| **Características (Features)** | 30 (ej. *mean radius*, *mean texture*) |
| **Clases** | Maligno (0) y Benigno (1) |
| **Distribución de Clases** | Maligno: 212 muestras (**37.3%**) |
| | Benigno: 357 muestras (**62.7%**) |



El conjunto de datos está ligeramente desbalanceado, pero la estratificación en el `train_test_split` asegura una representación equitativa de las clases.

### Pasos de Preprocesamiento

1.  **División de Datos:** El dataset se dividió en conjuntos de **Entrenamiento (80%)** y **Prueba (20%)** utilizando la función `train_test_split` con `stratify=y` para mantener la proporción de clases.
2.  **Escalado de Características:** Se aplicó el `StandardScaler` a todas las características. Esto es crucial para modelos basados en distancia (como KNN y SVM) y para la Regresión Logística, ya que asegura que todas las características contribuyan equitativamente, independientemente de su escala original.

## Modelos Evaluados

Se entrenaron y compararon cinco modelos de clasificación avanzados:

| Modelo | Tipo | Notas Clave |
| :--- | :--- | :--- |
| **KNeighbors Classifier (KNN)** | Basado en Instancias | Utiliza la distancia a los 5 vecinos más cercanos (`n_neighbors=5`). |
| **Random Forest** | Ensamblaje (Bagging) | 200 árboles de decisión (`n_estimators=200`) para reducir el sobreajuste. |
| **Support Vector Machine (SVM)** | Basado en Kernel | Utiliza un kernel RBF para mapear datos a un espacio de mayor dimensión. |
| **Logistic Regression** | Lineal | Modelo simple que predice la probabilidad de pertenencia a una clase. |
| **Gradient Boosting** | Ensamblaje (Boosting) | Construye secuencialmente árboles débiles, corrigiendo los errores del anterior. |

##  Resultados de la Evaluación

El rendimiento se evaluó utilizando la **Precisión (Accuracy)** en el conjunto de prueba, la **Validación Cruzada (5-Fold)** y la métrica clave: el **Área bajo la Curva ROC (AUC-ROC)**.

### Tabla de Rendimiento

| Modelo | Precisión (Test) | CV (Media ± Desv. Est.) | AUC-ROC |
| :--- | :--- | :--- | :--- |
| **KNN** | 96.49% | 97.58% ± 1.13% | 0.992 |
| **Random Forest** | 95.61% | 96.92% ± 1.83% | 0.989 |
| **SVM** | 98.25% | **97.80% ± 1.25%** | **0.999** |
| **Logistic Regression** | 98.25% | 97.58% ± 1.39% | 0.998 |
| **Gradient Boosting** | 95.61% | 96.70% ± 1.88% | 0.985 |

### Mejor Modelo

El modelo seleccionado como el mejor es el **Support Vector Machine (SVM)**, ya que alcanzó el AUC-ROC más alto.

* **SVM:** AUC-ROC = **0.999** (La capacidad de discriminación es casi perfecta).
* **Precisión en Test:** **98.25%**

## Análisis Gráfico de Resultados

Los resultados completos se visualizan en el gráfico `analisis_cancer_completo.png` generado por el script.

### 1. Curva ROC

La curva ROC (Receiver Operating Characteristic) mide la capacidad del modelo para distinguir entre clases. Un AUC cercano a 1.0 indica una excelente discriminación.



**Observación Clave:** Los modelos **SVM** y **Regresión Logística** muestran curvas casi perfectas, superando a los demás en la métrica más importante para diagnóstico.

### 2. Matriz de Confusión (Mejor Modelo: SVM)

La matriz de confusión del modelo SVM confirma su rendimiento casi perfecto, minimizando los errores de clasificación, especialmente los **Falsos Negativos (FN)** (un diagnóstico maligno perdido), lo cual es crítico en medicina.

| Etiqueta Real / Predicha | Maligno (0) | Benigno (1) |
| :--- | :--- | :--- |
| **Maligno (0)** | 40 (Verdaderos Negativos) | 1 (Falso Positivo) |
| **Benigno (1)** | 0 (Falso Negativo) | 73 (Verdaderos Positivos) |

**Error Crítico Cero:** El modelo SVM no cometió **ningún Falso Negativo** en el conjunto de prueba (0 casos malignos fueron predichos erróneamente como benignos), lo que lo hace excepcionalmente fiable desde una perspectiva clínica.

### 3. Importancia de Características (Random Forest)

Si el mejor modelo no proporciona importancia de características (como el SVM clásico), se analiza el siguiente mejor modelo de ensamblaje (Random Forest).

**Top 5 Características más Importantes (Random Forest):**

1.  **Worst perimeter**
2.  **Worst area**
3.  **Worst radius**
4.  **Mean concave points**
5.  **Mean perimeter**

## Análisis de Regresión Logística (Impacto de Características)

El análisis de coeficientes del modelo de Regresión Logística, después del escalado, revela las características que tienen el mayor impacto en la clasificación:

| Característica | Coeficiente | Impacto |
| :--- | :--- | :--- |
| **Worst fractal dimension** | 1.15 | Mayor probabilidad de ser **Benigno** (Clase 1) |
| **Smoothness error** | 1.04 | Mayor probabilidad de ser **Benigno** (Clase 1) |
| **Worst smoothness** | -0.76 | Mayor probabilidad de ser **Maligno** (Clase 0) |
| **Worst concavity** | -0.66 | Mayor probabilidad de ser **Maligno** (Clase 0) |
| **Worst perimeter** | -0.58 | Mayor probabilidad de ser **Maligno** (Clase 0) |

Los coeficientes negativos (e.g., *Worst smoothness*, *Worst concavity*, *Worst perimeter*) se asocian con la clase Maligna (0), mientras que los positivos (e.g., *Worst fractal dimension*) se asocian con la clase Benigna (1).

##  Conclusión Final

El análisis demuestra que los modelos **Support Vector Machine (SVM)** y **Logistic Regression** son los más adecuados para este problema de clasificación.

El modelo **SVM** se establece como el ganador con un **AUC-ROC de 0.999** y una **Precisión del 98.25%**, crucialmente, sin cometer **Falsos Negativos** en el conjunto de prueba. Esto lo convierte en una herramienta altamente confiable para la asistencia al diagnóstico clínico de cáncer de mama.

Para futuros pasos, se podría considerar la optimización de hiperparámetros de estos modelos ganadores para intentar alcanzar la precisión perfecta.
