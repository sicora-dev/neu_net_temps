import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Red Neuronal con Backpropagation implementada desde cero.
    
    Arquitectura:
    - Capa de entrada: n_inputs neuronas (9 en nuestro caso)
    - Capa oculta: n_hidden neuronas (16 por defecto)
    - Capa de salida: 1 neurona (predicción de temperatura)
    
    Funciones de activación:
    - Capa oculta: ReLU
    - Capa salida: Lineal (sin activación)
    """
    
    def __init__(self, n_inputs, n_hidden=16, learning_rate=0.001, random_seed=42):
        """
            n_inputs: Número de características de entrada (9 en nuestro caso)
            n_hidden: Número de neuronas en capa oculta (default: 16)
            learning_rate: Tasa de aprendizaje (default: 0.001)
            random_seed: Semilla para reproducibilidad
        """
        np.random.seed(random_seed)
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # ==============================================================
        # INICIALIZACIÓN DE PESOS Y SESGOS
        # ==============================================================
        # ¿Por qué inicializar con valores pequeños aleatorios?
        # - Si todos son 0: todas las neuronas aprenden lo mismo (simetría)
        # - Si son muy grandes: gradientes explotan
        # - Pequeños y aleatorios: cada neurona aprende patrones diferentes
        
        # Pesos capa entrada → oculta: matriz [n_inputs, n_hidden]
        # He initialization: escala óptima para ReLU
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
        
        # Sesgos capa oculta: vector [n_hidden]
        self.b1 = np.zeros((1, n_hidden))
        
        # Pesos capa oculta → salida: matriz [n_hidden, 1]
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        
        # Sesgo capa salida: escalar
        self.b2 = np.zeros((1, 1))
        
        # Historial de pérdidas para visualización
        self.loss_history = []
        
        print("🧠 Red Neuronal Inicializada")
        print(f"   Entrada: {n_inputs} → Oculta: {n_hidden} → Salida: 1")
        print(f"   Parámetros totales: {self._count_parameters()}")
        print(f"   Tasa de aprendizaje: {learning_rate}")
    
    def _count_parameters(self):
        return (self.W1.size + self.b1.size + 
                self.W2.size + self.b2.size)
    
    def relu(self, z):
        """
        ReLU: Rectified Linear Unit
        
        f(x) = max(0, x)
        
        ¿Por qué usarla?
        - Introduce no-linealidad (permite aprender patrones complejos)
        - Computacionalmente eficiente
        - Evita gradiente desvaneciente
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivada de ReLU
        
        f'(x) = 1 si x > 0, sino 0
        
        Necesitamos esto para backpropagation (calcular gradientes)
        """
        return (z > 0).astype(float)
    
    def forward(self, X):
        """
        Propagación hacia adelante: calcula la predicción.
        
        Flujo:
        X → [Pesos1] → z1 → [ReLU] → a1 → [Pesos2] → z2 → predicción
        
        Args:
            X: Datos de entrada [n_samples, n_inputs]
            
        Returns:
            prediction: Predicción de la red [n_samples, 1]
            cache: Diccionario con valores intermedios (para backward)
        """
        
        # Capa oculta
        # z1 = X · W1 + b1
        # Multiplicación matricial: [batch, inputs] × [inputs, hidden]
        z1 = np.dot(X, self.W1) + self.b1
        
        # Activación ReLU
        # a1 = ReLU(z1)
        a1 = self.relu(z1)
        
        # Capa de salida
        # z2 = a1 · W2 + b2
        z2 = np.dot(a1, self.W2) + self.b2
        
        # No aplicamos activación en la salida (regresión lineal)
        prediction = z2
        
        # Guardamos valores para usar en backward
        cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2
        }
        
        return prediction, cache
    
    def backward(self, y_true, cache):
        """
        Propagación hacia atrás: calcula gradientes usando la regla de la cadena.
        
        ¡ESTE ES EL CORAZÓN DEL ALGORITMO!
        
        La regla de la cadena nos dice cómo el error en la salida
        afecta a cada peso en la red.
        
        Args:
            y_true: Valores reales [n_samples, 1]
            cache: Valores intermedios del forward pass
            
        Returns:
            gradients: Diccionario con gradientes de cada parámetro
        """
        
        # Número de ejemplos
        m = y_true.shape[0]
        
        # Extraer valores del cache
        X = cache['X']
        z1 = cache['z1']
        a1 = cache['a1']
        z2 = cache['z2']
        
        # ==============================================================
        # PASO 1: Error en la capa de salida
        # ==============================================================
        # dz2 = predicción - real
        # Este es el "punto de partida" del error
        dz2 = z2 - y_true
        
        # ==============================================================
        # PASO 2: Gradientes de la capa de salida
        # ==============================================================
        # ¿Cómo afecta W2 al error?
        # dW2 = (1/m) × a1.T · dz2
        # Promediamos sobre todos los ejemplos (1/m)
        dW2 = (1/m) * np.dot(a1.T, dz2)
        
        # ¿Cómo afecta b2 al error?
        # db2 = (1/m) × suma(dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # ==============================================================
        # PASO 3: Propagar error a la capa oculta
        # ==============================================================
        # ¿Cuánto error viene de la capa siguiente?
        # da1 = dz2 · W2.T
        da1 = np.dot(dz2, self.W2.T)
        
        # Aplicar derivada de ReLU
        # dz1 = da1 ⊙ ReLU'(z1)
        # ⊙ = multiplicación elemento a elemento (Hadamard)
        # Solo las neuronas "activas" (z1 > 0) propagan el error
        dz1 = da1 * self.relu_derivative(z1)
        
        # ==============================================================
        # PASO 4: Gradientes de la capa oculta
        # ==============================================================
        # ¿Cómo afecta W1 al error?
        dW1 = (1/m) * np.dot(X.T, dz1)
        
        # ¿Cómo afecta b1 al error?
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Retornar todos los gradientes
        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return gradients
    
    def update_weights(self, gradients):
        """
        Actualiza pesos usando gradient descent.
        
        Regla: peso_nuevo = peso_viejo - learning_rate × gradiente
        
        ¿Por qué restamos?
        - El gradiente apunta hacia donde el error AUMENTA
        - Queremos ir en dirección opuesta (donde el error DISMINUYE)
        
        Args:
            gradients: Diccionario con gradientes calculados
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    
    def compute_loss(self, y_true, y_pred):
        """
        Calcula Mean Squared Error (Error Cuadrático Medio).
        
        MSE = (1/m) × Σ(predicción - real)²
        
        ¿Por qué MSE?
        - Penaliza errores grandes más que errores pequeños
        - Diferenciable (necesario para backpropagation)
        - Común en problemas de regresión
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            loss: Error promedio
        """
        m = y_true.shape[0]
        loss = (1/(2*m)) * np.sum((y_pred - y_true) ** 2)
        return loss
    
    # ==============================================================
    # ENTRENAMIENTO
    # ==============================================================
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=1000, batch_size=32, verbose=True):
        """
        Entrena la red neuronal.
        
        Args:
            X_train: Datos de entrenamiento [n_samples, n_inputs]
            y_train: Objetivos de entrenamiento [n_samples, 1]
            X_val: Datos de validación (opcional)
            y_val: Objetivos de validación (opcional)
            epochs: Número de épocas (pasadas completas por los datos)
            batch_size: Tamaño del mini-batch
            verbose: Si True, imprime progreso
        """
        n_samples = X_train.shape[0]
        
        # Asegurar que y_train tiene forma correcta
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        
        print(f"\n🏋️ Iniciando entrenamiento...")
        print(f"   Épocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Muestras de entrenamiento: {n_samples}")
        
        for epoch in range(epochs):
            # Mezclar datos al inicio de cada época
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch gradient descent
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                # Extraer mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred, cache = self.forward(X_batch)
                
                # Calcular pérdida
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass
                gradients = self.backward(y_batch, cache)
                
                # Actualizar pesos
                self.update_weights(gradients)
            
            # Pérdida promedio de la época
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            # Imprimir progreso cada 100 épocas
            if verbose and (epoch + 1) % 100 == 0:
                val_info = ""
                if X_val is not None:
                    y_val_pred, _ = self.forward(X_val)
                    val_loss = self.compute_loss(y_val, y_val_pred)
                    val_info = f" | Val Loss: {val_loss:.4f}"
                
                print(f"   Época {epoch+1}/{epochs} | Loss: {avg_loss:.4f}{val_info}")
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"   Pérdida final: {self.loss_history[-1]:.4f}")
    
    # ==============================================================
    # PREDICCIÓN
    # ==============================================================
    
    def predict(self, X):
        """
        Hace predicciones sobre nuevos datos.
        
        Args:
            X: Datos de entrada [n_samples, n_inputs]
            
        Returns:
            predictions: Predicciones [n_samples, 1]
        """
        predictions, _ = self.forward(X)
        return predictions
    
    # ==============================================================
    # EVALUACIÓN
    # ==============================================================
    
    def evaluate(self, X, y_true):
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            X: Datos de entrada
            y_true: Valores reales
            
        Returns:
            metrics: Diccionario con métricas de evaluación
        """
        y_pred = self.predict(X)
        
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        # MSE
        mse = np.mean((y_pred - y_true) ** 2)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_pred - y_true))
        
        # R² Score
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
    # ==============================================================
    # VISUALIZACIÓN
    # ==============================================================
    
    def plot_loss(self):
        """Grafica la curva de aprendizaje (pérdida vs época)."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Pérdida (MSE)', fontsize=12)
        plt.title('Curva de Aprendizaje', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ==============================================================
# EJEMPLO DE USO
# ==============================================================

if __name__ == "__main__":
    print("🧪 Prueba de la Red Neuronal\n")
    
    # Crear datos sintéticos para probar
    np.random.seed(42)
    n_samples = 1000
    n_inputs = 9
    
    # Generar datos: y = suma de entradas + ruido
    X = np.random.randn(n_samples, n_inputs)
    y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.1
    
    # Dividir en train/test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Crear y entrenar red
    nn = NeuralNetwork(n_inputs=n_inputs, n_hidden=16, learning_rate=0.01)
    nn.train(X_train, y_train, X_test, y_test, epochs=500, batch_size=32)
    
    # Evaluar
    metrics = nn.evaluate(X_test, y_test)
    print(f"\n📊 Métricas en test:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Visualizar
    nn.plot_loss()