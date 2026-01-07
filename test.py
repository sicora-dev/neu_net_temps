"""
Script de Prueba para el Modelo de Predicción de Temperatura

Este script:
1. Carga un modelo previamente entrenado
2. Permite hacer predicciones con datos nuevos
3. Muestra ejemplos de uso interactivo
"""

import numpy as np
import pickle
import os
from data_loader import WeatherDataLoader
from neural_network import NeuralNetwork


def load_model(filepath):
    """
    Carga un modelo guardado.
    
    Args:
        filepath: Ruta al archivo .pkl del modelo
        
    Returns:
        nn: Red neuronal cargada
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el modelo en: {filepath}")
    
    with open(filepath, 'rb') as f:
        nn = pickle.load(f)
    
    print(f"✅ Modelo cargado desde: {filepath}")
    print(f"   Arquitectura: {nn.n_inputs} → {nn.n_hidden} → 1")
    print(f"   Épocas entrenadas: {len(nn.loss_history)}")
    
    return nn


def predict_with_manual_input(nn, scaler):
    """
    Permite hacer predicciones ingresando datos manualmente.
    
    Args:
        nn: Red neuronal entrenada
        scaler: Normalizador usado en entrenamiento
    """
    print("\n" + "="*60)
    print("🔮 PREDICCIÓN INTERACTIVA")
    print("="*60)
    
    print("\nIngresa los datos de los últimos 3 meses:")
    print("(T2M = temperatura promedio, MAX = temperatura máxima, MIN = temperatura mínima)")
    
    features = []
    
    for mes in range(1, 4):
        print(f"\n--- Mes {mes} ---")
        
        t2m = float(input(f"  T2M promedio (°C): "))
        t2m_max = float(input(f"  T2M_MAX (°C): "))
        t2m_min = float(input(f"  T2M_MIN (°C): "))
        
        features.extend([t2m, t2m_max, t2m_min])
    
    # Convertir a array y normalizar
    X = np.array(features).reshape(1, -1)
    X_normalized = scaler.transform(X)
    
    # Predecir
    prediction = nn.predict(X_normalized)
    
    print("\n" + "-"*60)
    print(f"🌡️  PREDICCIÓN: {prediction[0][0]:.2f}°C")
    print("-"*60)
    
    # Mostrar contexto
    avg_temp = np.mean([features[i] for i in range(0, 9, 3)])
    print(f"\n📊 Contexto:")
    print(f"   Temperatura promedio de los 3 meses: {avg_temp:.2f}°C")
    print(f"   Predicción para el mes siguiente: {prediction[0][0]:.2f}°C")
    
    if prediction[0][0] > avg_temp:
        print(f"   → La temperatura aumentará ~{prediction[0][0] - avg_temp:.2f}°C")
    else:
        print(f"   → La temperatura disminuirá ~{avg_temp - prediction[0][0]:.2f}°C")


def predict_from_dataset(nn, data_path, n_examples=5):
    """
    Hace predicciones sobre ejemplos del dataset.
    
    Args:
        nn: Red neuronal entrenada
        data_path: Ruta al dataset
        n_examples: Número de ejemplos a mostrar
    """
    print("\n" + "="*60)
    print("📊 PREDICCIONES SOBRE DATASET REAL")
    print("="*60)
    
    # Cargar datos
    loader = WeatherDataLoader(filepath=data_path, n_months_history=3)
    X_train, X_test, y_train, y_test = loader.load_and_prepare()
    
    # Hacer predicciones en test set
    y_pred = nn.predict(X_test)
    
    # Calcular métricas
    metrics = nn.evaluate(X_test, y_test)
    
    print(f"\n📈 Métricas en Test Set:")
    print(f"   RMSE: {metrics['RMSE']:.2f}°C")
    print(f"   MAE:  {metrics['MAE']:.2f}°C")
    print(f"   R²:   {metrics['R2']:.4f}")
    
    # Mostrar ejemplos individuales
    print(f"\n🔍 Ejemplos de Predicciones (primeros {n_examples}):")
    print("-"*60)
    
    for i in range(min(n_examples, len(X_test))):
        real = y_test[i]
        pred = y_pred[i][0]
        error = pred - real
        
        print(f"\nEjemplo {i+1}:")
        print(f"   Real:      {real:.2f}°C")
        print(f"   Predicho:  {pred:.2f}°C")
        print(f"   Error:     {error:+.2f}°C", end="")
        
        # Indicador visual del error
        if abs(error) < 1.0:
            print(" ✅ (excelente)")
        elif abs(error) < 2.0:
            print(" 👍 (bueno)")
        elif abs(error) < 3.0:
            print(" ⚠️  (aceptable)")
        else:
            print(" ❌ (mejorable)")


def analyze_worst_predictions(nn, data_path, n_worst=5):
    """
    Analiza las peores predicciones para entender errores del modelo.
    
    Args:
        nn: Red neuronal entrenada
        data_path: Ruta al dataset
        n_worst: Número de peores casos a mostrar
    """
    print("\n" + "="*60)
    print("🔍 ANÁLISIS DE PEORES PREDICCIONES")
    print("="*60)
    
    # Cargar datos
    loader = WeatherDataLoader(filepath=data_path, n_months_history=3)
    X_train, X_test, y_train, y_test = loader.load_and_prepare()
    
    # Predicciones
    y_pred = nn.predict(X_test)
    
    # Calcular errores absolutos
    errors = np.abs((y_pred - y_test.reshape(-1, 1))).flatten()
    
    # Encontrar los peores casos
    worst_indices = np.argsort(errors)[-n_worst:][::-1]
    
    print(f"\n❌ Top {n_worst} Peores Predicciones:")
    print("-"*60)
    
    for rank, idx in enumerate(worst_indices, 1):
        real = y_test[idx]
        pred = y_pred[idx][0]
        error = pred - real
        
        print(f"\n#{rank} - Error absoluto: {abs(error):.2f}°C")
        print(f"   Real:      {real:.2f}°C")
        print(f"   Predicho:  {pred:.2f}°C")
        print(f"   Error:     {error:+.2f}°C")
        
        # Datos de entrada (desnormalizados estarían mejor, pero es complejo)
        print(f"   Entrada: {X_test[idx][:3]} ... (primeros 3 valores)")


def compare_with_simple_baseline(nn, data_path):
    """
    Compara el modelo con estrategias simples (baseline).
    
    Args:
        nn: Red neuronal entrenada
        data_path: Ruta al dataset
    """
    print("\n" + "="*60)
    print("⚖️  COMPARACIÓN CON BASELINES")
    print("="*60)
    
    # Cargar datos
    loader = WeatherDataLoader(filepath=data_path, n_months_history=3)
    X_train, X_test, y_train, y_test = loader.load_and_prepare()
    
    # Predicción del modelo
    y_pred = nn.predict(X_test)
    model_mse = np.mean((y_pred.flatten() - y_test) ** 2)
    
    # Baseline 1: Siempre predecir el promedio
    baseline_mean = np.mean(y_train)
    baseline1_mse = np.mean((baseline_mean - y_test) ** 2)
    
    # Baseline 2: Predecir el último valor conocido (persistencia)
    # Usamos el último valor de cada secuencia de entrada
    last_values = X_train[:, 0]  # Primera característica (T2M del último mes)
    baseline2_mse = np.mean((np.mean(last_values) - y_test) ** 2)
    
    print("\n📊 Comparación de MSE:")
    print(f"\n   1️⃣  Baseline (predecir promedio):")
    print(f"       MSE = {baseline1_mse:.4f}")
    
    print(f"\n   2️⃣  Baseline (persistencia):")
    print(f"       MSE = {baseline2_mse:.4f}")
    
    print(f"\n   3️⃣  Nuestro Modelo (Red Neuronal):")
    print(f"       MSE = {model_mse:.4f}")
    
    # Mejora porcentual
    improvement1 = ((baseline1_mse - model_mse) / baseline1_mse) * 100
    improvement2 = ((baseline2_mse - model_mse) / baseline2_mse) * 100
    
    print(f"\n🎯 Mejora respecto a baselines:")
    print(f"   vs Promedio: {improvement1:+.2f}%")
    print(f"   vs Persistencia: {improvement2:+.2f}%")
    
    if improvement1 > 0 and improvement2 > 0:
        print(f"\n✅ ¡El modelo supera ambos baselines!")
    elif improvement1 > 0:
        print(f"\n⚠️  El modelo solo supera al baseline de promedio")
    else:
        print(f"\n❌ El modelo no supera los baselines. Necesita mejora.")


def main():
    """
    Función principal con menú interactivo.
    """
    print("="*60)
    print("🧪 PRUEBA DEL MODELO DE PREDICCIÓN DE TEMPERATURA")
    print("="*60)
    
    # Buscar modelos disponibles
    models = [f for f in os.listdir('.') if f.endswith('.pkl') and f.startswith('modelo_temperatura')]
    
    if not models:
        print("\n❌ No se encontraron modelos entrenados.")
        print("💡 Primero ejecuta 'train.py' para entrenar un modelo.")
        return
    
    # Seleccionar modelo
    print(f"\n📂 Modelos disponibles:")
    for i, model in enumerate(models, 1):
        size = os.path.getsize(model) / 1024
        print(f"   {i}. {model} ({size:.2f} KB)")
    
    if len(models) == 1:
        selected_model = models[0]
        print(f"\n→ Usando: {selected_model}")
    else:
        choice = int(input(f"\nSelecciona un modelo (1-{len(models)}): "))
        selected_model = models[choice - 1]
    
    # Cargar modelo
    nn = load_model(selected_model)
    
    # Menú de opciones
    while True:
        print("\n" + "="*60)
        print("📋 MENÚ DE OPCIONES")
        print("="*60)
        print("\n1. Predicción interactiva (ingresar datos manualmente)")
        print("2. Predicciones sobre dataset real")
        print("3. Analizar peores predicciones")
        print("4. Comparar con baselines")
        print("5. Salir")
        
        choice = input("\nSelecciona una opción (1-5): ")
        
        if choice == '1':
            # Necesitamos el scaler del DataLoader
            data_path = 'southamerica_0_regional_monthly.csv'
            if not os.path.exists(data_path):
                print(f"\n❌ No se encontró el dataset: {data_path}")
                continue
            
            loader = WeatherDataLoader(filepath=data_path, n_months_history=3)
            X_train, X_test, y_train, y_test = loader.load_and_prepare()
            
            predict_with_manual_input(nn, loader.scaler)
            
        elif choice == '2':
            data_path = 'southamerica_0_regional_monthly.csv'
            if not os.path.exists(data_path):
                print(f"\n❌ No se encontró el dataset: {data_path}")
                continue
            predict_from_dataset(nn, data_path)
            
        elif choice == '3':
            data_path = 'southamerica_0_regional_monthly.csv'
            if not os.path.exists(data_path):
                print(f"\n❌ No se encontró el dataset: {data_path}")
                continue
            analyze_worst_predictions(nn, data_path)
            
        elif choice == '4':
            data_path = 'southamerica_0_regional_monthly.csv'
            if not os.path.exists(data_path):
                print(f"\n❌ No se encontró el dataset: {data_path}")
                continue
            compare_with_simple_baseline(nn, data_path)
            
        elif choice == '5':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    main()