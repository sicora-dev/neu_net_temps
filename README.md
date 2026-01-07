# 🌡️ Predicción de Temperatura con Red Neuronal desde Cero

Implementación completa de una **Red Neuronal con Backpropagation** para predecir temperaturas mensuales usando datos meteorológicos de NASA POWER.

---

## 📋 Descripción del Proyecto

Este proyecto implementa una red neuronal **completamente desde cero** (sin usar librerías de deep learning como TensorFlow o PyTorch) para predecir la temperatura promedio del mes siguiente basándose en datos históricos de temperatura.

### ✨ Características

- ✅ Backpropagation implementado manualmente
- ✅ Utiliza datos reales de NASA POWER (Sudamérica, 1984-2022)
- ✅ Incluye tres variables: T2M, T2M_MAX, T2M_MIN
- ✅ Visualizaciones completas de resultados
- ✅ Comparación con baselines
- ✅ Análisis de errores
- ✅ Interfaz interactiva para predicciones

---

## 🏗️ Arquitectura de la Red

```
ENTRADA (9 neuronas)
  ↓
[T2M, MAX, MIN] × 3 meses
  ↓
CAPA OCULTA (16 neuronas)
  ↓
Activación: ReLU
  ↓
CAPA SALIDA (1 neurona)
  ↓
Predicción: T2M mes siguiente
```

### Componentes Implementados

1. **Forward Propagation**: Cálculo de predicciones
2. **Backward Propagation**: Cálculo de gradientes usando regla de la cadena
3. **Gradient Descent**: Actualización de pesos
4. **Mini-Batch Training**: Entrenamiento eficiente por lotes
5. **Normalización**: Estandarización de datos

---

## 📂 Estructura del Proyecto

```
proyecto_backpropagation/
│
├── data_loader.py          # Carga y prepara datos NASA POWER
├── neural_network.py       # Red neuronal con backpropagation
├── train.py                # Script de entrenamiento
├── test.py                 # Script de prueba/predicción
├── README.md               # Este archivo
│
├── southamerica_0_regional_monthly.csv  # Dataset (descargar)
│
└── resultados/             # Gráficas generadas (creado automáticamente)
    ├── curva_aprendizaje_y_predicciones.png
    └── analisis_errores.png
```

---

## 🚀 Instalación

### Requisitos

- Python 3.7+
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

### Instalar Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 📥 Descargar el Dataset

1. Ve a: [https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/](https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/)
2. Descarga: `csvs/southamerica/southamerica_0_regional_monthly.csv`
3. Coloca el archivo en la carpeta del proyecto

**Alternativa (línea de comandos):**
```bash
wget https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/resolve/main/csvs/southamerica/southamerica_0_regional_monthly.csv
```

---

## 🎯 Uso

### 1. Entrenar el Modelo

```bash
python train.py
```

**Esto hará:**
- ✅ Cargará y preparará los datos
- ✅ Creará la red neuronal
- ✅ Entrenará por 1000 épocas
- ✅ Generará visualizaciones
- ✅ Guardará el modelo entrenado

**Resultado esperado:**
```
📊 MÉTRICAS DE EVALUACIÓN
====================================================
📏 Raíz del Error Cuadrático Medio (RMSE):
   2.34°C
   → En promedio, nos equivocamos ±2.34°C

⭐ Coeficiente de Determinación (R²):
   0.8523
   → Bueno. El modelo captura la mayoría de patrones
```

### 2. Probar el Modelo

```bash
python test.py
```

**Opciones disponibles:**

#### a) Predicción Interactiva
Ingresa datos manualmente:
```
Mes 1:
  T2M promedio: 20
  T2M_MAX: 25
  T2M_MIN: 15
Mes 2: ...
```

#### b) Predicciones sobre Dataset
Ve ejemplos reales de predicciones.

#### c) Análisis de Errores
Identifica dónde falla más el modelo.

#### d) Comparación con Baselines
Compara vs métodos simples.

---

## 🧠 Cómo Funciona (Backpropagation)

### Algoritmo Simplificado

```python
for cada época:
    for cada mini-batch:
        # 1. FORWARD PASS
        predicción = calcular_salida(entrada)
        
        # 2. CALCULAR ERROR
        error = predicción - valor_real
        
        # 3. BACKWARD PASS (Backpropagation)
        gradiente_salida = calcular_gradiente_capa_salida(error)
        gradiente_oculta = propagar_error_hacia_atras(gradiente_salida)
        
        # 4. ACTUALIZAR PESOS
        pesos -= learning_rate × gradiente
```

### Fórmulas Clave

**Forward Pass:**
```
z1 = X · W1 + b1
a1 = ReLU(z1)
z2 = a1 · W2 + b2
predicción = z2
```

**Backward Pass:**
```
dz2 = predicción - y_real
dW2 = a1^T · dz2
da1 = dz2 · W2^T
dz1 = da1 ⊙ ReLU'(z1)
dW1 = X^T · dz1
```

**Actualización:**
```
W = W - α × dW
```
donde α = learning rate

---

## ⚙️ Configuración e Hiperparámetros

En `train.py` puedes ajustar:

```python
HIDDEN_NEURONS = 16      # Neuronas en capa oculta
LEARNING_RATE = 0.001    # Tasa de aprendizaje
EPOCHS = 1000            # Número de épocas
BATCH_SIZE = 32          # Tamaño del mini-batch
N_MONTHS_HISTORY = 3     # Meses de historia
```

### Guía de Ajuste

| Problema | Solución |
|----------|----------|
| Pérdida muy alta | ↑ Aumentar neuronas ocultas<br>↑ Aumentar épocas |
| No converge | ↓ Reducir learning rate |
| Converge muy lento | ↑ Aumentar learning rate |
| Overfitting | ↓ Reducir neuronas ocultas<br>Agregar más datos |

---

## 📊 Interpretación de Resultados

### Métricas

| Métrica | Significado | Valor Bueno |
|---------|-------------|-------------|
| **RMSE** | Error típico en °C | < 3.0°C |
| **MAE** | Error absoluto promedio | < 2.5°C |
| **R²** | % de variación explicada | > 0.7 |

### Gráficas

#### 1. Curva de Aprendizaje
- **Descendente**: ✅ El modelo aprende
- **Plana muy alta**: ❌ No puede aprender (underfitting)
- **Oscilatoria**: ⚠️ Learning rate muy alto

#### 2. Predicciones vs Reales
- **Puntos cerca de línea**: ✅ Buenas predicciones
- **Puntos dispersos**: ❌ Predicciones inconsistentes
- **Patrón sistemático**: ⚠️ Sesgo en el modelo

#### 3. Distribución de Errores
- **Centrada en 0**: ✅ Sin sesgo
- **Forma de campana**: ✅ Errores aleatorios
- **Desplazada**: ❌ Modelo sobre/subestima

---

## 🎓 Conceptos Aprendidos

### Matemáticas Implementadas

- ✅ Multiplicación de matrices
- ✅ Regla de la cadena (cálculo)
- ✅ Derivadas parciales
- ✅ Gradient Descent
- ✅ Función ReLU y su derivada

### Machine Learning

- ✅ Forward/Backward propagation
- ✅ Mini-batch training
- ✅ Normalización de datos
- ✅ Train/Test split
- ✅ Métricas de evaluación
- ✅ Baselines de comparación

### Buenas Prácticas

- ✅ Código modular y documentado
- ✅ Manejo de errores
- ✅ Visualizaciones informativas
- ✅ Reproducibilidad (random_seed)

---

## 🔧 Solución de Problemas

### Error: "No se encontró el archivo"
```
❌ ERROR: No se encontró el archivo southamerica_0_regional_monthly.csv
```
**Solución:** Descarga el dataset de HuggingFace (ver sección "Descargar el Dataset")

### Error: "No such file or directory: 'modelo_temperatura.pkl'"
**Solución:** Primero ejecuta `train.py` para entrenar y guardar un modelo

### Pérdida no disminuye
**Posibles causas:**
- Learning rate muy alto → Reducir a 0.0001
- Datos no normalizados → Verificar que DataLoader normaliza
- Arquitectura inadecuada → Probar con más/menos neuronas

### Predicciones siempre iguales
**Posibles causas:**
- Pesos inicializados en cero → El código ya usa inicialización aleatoria
- Learning rate muy bajo → Aumentar a 0.01
- Convergió a mínimo local → Reiniciar con diferente random_seed

---

## 📚 Referencias

### Dataset
- **NASA POWER**: [https://power.larc.nasa.gov/](https://power.larc.nasa.gov/)
- **HuggingFace Dataset**: [https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather](https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather)

### Teoría
- **Backpropagation**: Rumelhart, Hinton & Williams (1986)
- **ReLU**: Nair & Hinton (2010)
- **Batch Normalization**: Ioffe & Szegedy (2015)

### Librerías
- **NumPy**: [https://numpy.org/](https://numpy.org/)
- **Pandas**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **Matplotlib**: [https://matplotlib.org/](https://matplotlib.org/)

---

## 🚀 Próximos Pasos

### Mejoras Posibles

1. **Agregar más características**
   - Precipitación
   - Humedad
   - Presión atmosférica

2. **Arquitectura más compleja**
   - Múltiples capas ocultas
   - Dropout para regularización
   - Batch normalization

3. **Optimización avanzada**
   - Adam optimizer
   - Learning rate decay
   - Early stopping

4. **Validación cruzada**
   - K-fold cross-validation
   - Time series split

5. **Comparación con librerías**
   - Implementar en TensorFlow
   - Implementar en PyTorch
   - Comparar rendimiento

---

## 📝 Notas

- **Tiempo de entrenamiento**: ~2-5 minutos en CPU moderna
- **Precisión esperada**: RMSE entre 2-4°C
- **Dataset size**: ~38 MB
- **Modelo guardado**: ~50-100 KB

---

## 👨‍💻 Autor

Proyecto educativo para aprender backpropagation desde cero.

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---

## 🙏 Agradecimientos

- NASA POWER por los datos meteorológicos
- HuggingFace por hospedar el dataset
- Comunidad de Machine Learning por recursos educativos

---

**¡Happy Learning! 🎓🚀**