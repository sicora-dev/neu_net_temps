"""
Script de Entrenamiento para Red Neuronal de Predicción Meteorológica

Este script:
1. Carga datos meteorológicos de NASA POWER
2. Prepara los datos (normalización, train/test split)
3. Entrena una red neuronal con backpropagation
4. Evalúa el modelo
5. Visualiza resultados
6. Guarda el modelo entrenado
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# Importar nuestras clases personalizadas
from data_loader import WeatherDataLoader
from neural_network import NeuralNetwork


def plot_results(nn, X_test, y_test, save_path='resultados'):
    """
    Crea visualizaciones completas de los resultados.
    
    Args:
        nn: Red neuronal entrenada
        X_test: Datos de test
        y_test: Valores reales de test
        save_path: Carpeta donde guardar las gráficas
    """
    # Crear carpeta si no existe
    os.makedirs(save_path, exist_ok=True)
    
    # Hacer predicciones
    y_pred = nn.predict(X_test)
    
    # ==============================================================
    # GRÁFICA 1: Curva de Aprendizaje
    # ==============================================================
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn.loss_history, linewidth=2, color='#2E86AB')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida (MSE)', fontsize=12)
    plt.title('Curva de Aprendizaje', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Marcar pérdida inicial y final
    plt.scatter([0, len(nn.loss_history)-1], 
                [nn.loss_history[0], nn.loss_history[-1]], 
                color='red', s=100, zorder=5)
    plt.text(0, nn.loss_history[0], 
             f' Inicial: {nn.loss_history[0]:.4f}', 
             verticalalignment='bottom', fontsize=9)
    plt.text(len(nn.loss_history)-1, nn.loss_history[-1], 
             f' Final: {nn.loss_history[-1]:.4f}', 
             verticalalignment='top', fontsize=9)
    
    # ==============================================================
    # GRÁFICA 2: Predicciones vs Reales
    # ==============================================================
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, color='#A23B72')
    
    # Línea de predicción perfecta (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Predicción perfecta')
    
    plt.xlabel('Temperatura Real (°C)', fontsize=12)
    plt.ylabel('Temperatura Predicha (°C)', fontsize=12)
    plt.title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/curva_aprendizaje_y_predicciones.png', dpi=300)
    print(f"   ✅ Guardado: {save_path}/curva_aprendizaje_y_predicciones.png")
    
    # ==============================================================
    # GRÁFICA 3: Distribución de Errores
    # ==============================================================
    plt.figure(figsize=(12, 5))
    
    errors = (y_pred - y_test).flatten()
    
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    plt.xlabel('Error de Predicción (°C)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Errores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Estadísticas del error
    plt.text(0.02, 0.98, 
             f'Media: {errors.mean():.3f}°C\n'
             f'Std: {errors.std():.3f}°C\n'
             f'Min: {errors.min():.3f}°C\n'
             f'Max: {errors.max():.3f}°C',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # ==============================================================
    # GRÁFICA 4: Errores a lo largo de las muestras
    # ==============================================================
    plt.subplot(1, 2, 2)
    plt.plot(np.abs(errors), linewidth=1, alpha=0.7, color='#C73E1D')
    plt.axhline(y=np.mean(np.abs(errors)), color='blue', 
                linestyle='--', linewidth=2, 
                label=f'MAE promedio: {np.mean(np.abs(errors)):.3f}°C')
    plt.xlabel('Muestra de Test', fontsize=12)
    plt.ylabel('Error Absoluto (°C)', fontsize=12)
    plt.title('Error Absoluto por Muestra', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/analisis_errores.png', dpi=300)
    print(f"   ✅ Guardado: {save_path}/analisis_errores.png")
    
    plt.show()


def print_detailed_metrics(metrics):
    """
    Imprime métricas de evaluación de forma legible.
    
    Args:
        metrics: Diccionario con métricas
    """
    print("\n" + "="*60)
    print("📊 MÉTRICAS DE EVALUACIÓN")
    print("="*60)
    
    print(f"\n🎯 Error Cuadrático Medio (MSE):")
    print(f"   {metrics['MSE']:.4f}")
    print(f"   → Mide el promedio de los errores al cuadrado")
    
    print(f"\n📏 Raíz del Error Cuadrático Medio (RMSE):")
    print(f"   {metrics['RMSE']:.4f}°C")
    print(f"   → En promedio, nos equivocamos ±{metrics['RMSE']:.2f}°C")
    
    print(f"\n📐 Error Absoluto Medio (MAE):")
    print(f"   {metrics['MAE']:.4f}°C")
    print(f"   → Error típico sin considerar dirección")
    
    print(f"\n⭐ Coeficiente de Determinación (R²):")
    print(f"   {metrics['R2']:.4f}")
    if metrics['R2'] >= 0.9:
        interpretation = "Excelente! El modelo explica >90% de la variación"
    elif metrics['R2'] >= 0.7:
        interpretation = "Bueno. El modelo captura la mayoría de patrones"
    elif metrics['R2'] >= 0.5:
        interpretation = "Aceptable. Hay margen de mejora"
    else:
        interpretation = "Bajo. El modelo necesita ajustes"
    print(f"   → {interpretation}")
    
    print("\n" + "="*60)


def save_model(nn, filepath='modelo_temperatura.pkl'):
    """
    Guarda el modelo entrenado en un archivo.
    
    Args:
        nn: Red neuronal entrenada
        filepath: Ruta donde guardar
    """
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)
    print(f"\n💾 Modelo guardado en: {filepath}")
    print(f"   Tamaño del archivo: {os.path.getsize(filepath) / 1024:.2f} KB")


def load_model(filepath='modelo_temperatura.pkl'):
    """
    Carga un modelo previamente entrenado.
    
    Args:
        filepath: Ruta del modelo guardado
        
    Returns:
        nn: Red neuronal cargada
    """
    with open(filepath, 'rb') as f:
        nn = pickle.load(f)
    print(f"\n📂 Modelo cargado desde: {filepath}")
    return nn


def main():
    """
    Función principal que ejecuta todo el pipeline de entrenamiento.
    """
    print("="*60)
    print("🌡️  PREDICCIÓN DE TEMPERATURA CON RED NEURONAL")
    print("="*60)
    print(f"\nFecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==============================================================
    # PASO 1: Configuración
    # ==============================================================
    print("\n" + "-"*60)
    print("⚙️  PASO 1: Configuración")
    print("-"*60)
    
    # Ruta al dataset
    DATA_PATH = 'southamerica_0_regional_monthly.csv'
    
    # Hiperparámetros
    HIDDEN_NEURONS = 16      # Neuronas en capa oculta
    LEARNING_RATE = 0.001    # Tasa de aprendizaje
    EPOCHS = 1000            # Número de épocas
    BATCH_SIZE = 32          # Tamaño del mini-batch
    N_MONTHS_HISTORY = 3     # Meses de historia para predecir
    
    print(f"\n📁 Dataset: {DATA_PATH}")
    print(f"\n🔧 Hiperparámetros:")
    print(f"   - Neuronas ocultas: {HIDDEN_NEURONS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Épocas: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Meses de historia: {N_MONTHS_HISTORY}")
    
    # ==============================================================
    # PASO 2: Cargar y Preparar Datos
    # ==============================================================
    print("\n" + "-"*60)
    print("📊 PASO 2: Carga y Preparación de Datos")
    print("-"*60)
    
    try:
        loader = WeatherDataLoader(
            filepath=DATA_PATH,
            n_months_history=N_MONTHS_HISTORY
        )
        
        X_train, X_test, y_train, y_test = loader.load_and_prepare()
        
        print(f"\n✅ Datos preparados exitosamente!")
        print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   - Test: {X_test.shape[0]} muestras")
        print(f"   - Características: {X_train.shape[1]}")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo {DATA_PATH}")
        print(f"\n💡 Solución:")
        print(f"   1. Descarga el dataset de:")
        print(f"      https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/")
        print(f"   2. Coloca el archivo en la misma carpeta que este script")
        print(f"   3. Asegúrate de que se llame: {DATA_PATH}")
        return
    except Exception as e:
        print(f"\n❌ ERROR al cargar datos: {e}")
        return
    
    # ==============================================================
    # PASO 3: Crear Red Neuronal
    # ==============================================================
    print("\n" + "-"*60)
    print("🧠 PASO 3: Creación de la Red Neuronal")
    print("-"*60)
    
    nn = NeuralNetwork(
        n_inputs=X_train.shape[1],
        n_hidden=HIDDEN_NEURONS,
        learning_rate=LEARNING_RATE
    )
    
    # ==============================================================
    # PASO 4: Entrenamiento
    # ==============================================================
    print("\n" + "-"*60)
    print("🏋️  PASO 4: Entrenamiento")
    print("-"*60)
    
    nn.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    # ==============================================================
    # PASO 5: Evaluación
    # ==============================================================
    print("\n" + "-"*60)
    print("📈 PASO 5: Evaluación del Modelo")
    print("-"*60)
    
    # Evaluar en conjunto de test
    metrics = nn.evaluate(X_test, y_test)
    print_detailed_metrics(metrics)
    
    # Comparar con un baseline simple
    baseline_pred = np.mean(y_train)  # Siempre predecir el promedio
    baseline_mse = np.mean((y_test.flatten() - baseline_pred) ** 2)
    print(f"\n🔍 Comparación con Baseline:")
    print(f"   Baseline (siempre predecir promedio):")
    print(f"   - MSE: {baseline_mse:.4f}")
    print(f"   Nuestra red:")
    print(f"   - MSE: {metrics['MSE']:.4f}")
    improvement = ((baseline_mse - metrics['MSE']) / baseline_mse) * 100
    print(f"   → Mejora: {improvement:.2f}%")
    
    # ==============================================================
    # PASO 6: Visualización
    # ==============================================================
    print("\n" + "-"*60)
    print("📊 PASO 6: Generación de Visualizaciones")
    print("-"*60)
    
    plot_results(nn, X_test, y_test)
    
    # ==============================================================
    # PASO 7: Guardar Modelo
    # ==============================================================
    print("\n" + "-"*60)
    print("💾 PASO 7: Guardado del Modelo")
    print("-"*60)
    
    model_filename = f'modelo_temperatura_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    save_model(nn, model_filename)
    
    # ==============================================================
    # RESUMEN FINAL
    # ==============================================================
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    
    print(f"\n📋 Resumen:")
    print(f"   ✅ Datos cargados y procesados")
    print(f"   ✅ Red neuronal entrenada ({EPOCHS} épocas)")
    print(f"   ✅ Modelo evaluado (RMSE: {metrics['RMSE']:.2f}°C)")
    print(f"   ✅ Visualizaciones generadas")
    print(f"   ✅ Modelo guardado: {model_filename}")
    
    print(f"\n🎯 Próximos pasos:")
    print(f"   1. Revisa las gráficas generadas en la carpeta 'resultados/'")
    print(f"   2. Si los resultados no son buenos, ajusta hiperparámetros")
    print(f"   3. Usa el modelo guardado para hacer predicciones")
    
    print("\n" + "="*60)


# ==============================================================
# PUNTO DE ENTRADA
# ==============================================================

if __name__ == "__main__":
    """
    Este bloque se ejecuta solo si corremos este archivo directamente.
    No se ejecuta si importamos funciones desde otro script.
    """
    main()