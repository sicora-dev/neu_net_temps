import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WeatherDataLoader:
    """
    
    Este cargador usa las tres variables de temperatura como entrada:
    - T2M: Temperatura promedio
    - T2M_MAX: Temperatura máxima
    - T2M_MIN: Temperatura mínima
    
    Y predice T2M del mes siguiente.
    """
    
    def __init__(self, filepath, n_months_history=3):
        self.filepath = filepath
        self.n_months = n_months_history
        self.scaler = StandardScaler()  # Para normalizar datos, sino el modelo pensaria que si temperatura es 20 y humedad 80, la humedad es 4 veces mas importante.

        self.temp_variables = ['T2M', 'T2M_MAX', 'T2M_MIN']
        
    def load_and_prepare(self):
        print("📥 Cargando datos...")
        df = pd.read_csv(self.filepath)
        
        print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"\n📊 Primeras columnas: {list(df.columns[:10])}")
        
        # Identificar columnas para cada variable de temperatura
        temp_data = {}
        for var in self.temp_variables:
            cols = [col for col in df.columns if col.startswith(var + '_') and 
                    col[len(var)+1:].isdigit() and 
                    1 <= int(col[len(var)+1:]) <= 12]
            print(cols)
            if cols:
                temp_data[var] = df[cols].values
                print(f"✅ {var}: {len(cols)} columnas encontradas")
            else:
                print(f"⚠️  {var}: No encontrado, se omitirá")
        
        if not temp_data:
            raise ValueError("No se encontraron variables de temperatura en el dataset")
        
        print(f"\n🎯 Estrategia:")
        print(f"   - Entrada: {list(temp_data.keys())} de {self.n_months} meses")
        print(f"   - Salida: Predecir T2M del mes siguiente")

        X, y = self._create_sequences(temp_data)
        
        # Dividir en train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n📦 Datos preparados:")
        print(f"   Train: {X_train.shape[0]} muestras")
        print(f"   Test: {X_test.shape[0]} muestras")
        print(f"   Características por muestra: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def _create_sequences(self, temp_data):
        X_list = []
        y_list = []
        
        # Usaremos T2M como variable objetivo (la que predeciremos)
        if 'T2M' not in temp_data:
            raise ValueError("T2M debe estar presente para predecir")
        
        target_data = temp_data['T2M']
        n_rows, n_months_total = target_data.shape

        for row_idx in range(n_rows):
            for month_start in range(n_months_total - self.n_months):

                features = []

                for month_offset in range(self.n_months):
                    current_month = month_start + month_offset

                    for var_name, var_data in temp_data.items():
                        value = var_data[row_idx, current_month]
                        features.append(value)

                target_month = month_start + self.n_months
                target_value = target_data[row_idx, target_month]

                if not (np.isnan(features).any() or np.isnan(target_value)):
                    X_list.append(features)
                    y_list.append(target_value)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        n_features = len(temp_data) * self.n_months
        
        print(f"\n🔄 Secuencias creadas:")
        print(f"   Total de secuencias: {len(X)}")
        print(f"   Características por muestra: {n_features}")
        print(f"   ({len(temp_data)} variables × {self.n_months} meses)")
        print(f"\n   Ejemplo entrada: {X[0]}")
        print(f"   Ejemplo salida (T2M siguiente): {y[0]:.2f}°C")
        
        return X, y


if __name__ == "__main__":
    # Este bloque solo se ejecuta si corres este archivo directamente
    print("🧪 Prueba del DataLoader\n")

    filepath = "southamerica_0_regional_monthly.csv"
    
    try:
        loader = WeatherDataLoader(
            filepath=filepath,
            n_months_history=3
        )
        
        X_train, X_test, y_train, y_test = loader.load_and_prepare()
        
        print("\n✅ ¡DataLoader funcionando correctamente!")
        print(f"\n📊 Resumen final:")
        print(f"   - Usamos T2M, T2M_MAX, T2M_MIN como entrada")
        print(f"   - Predecimos T2M del mes siguiente")
        print(f"   - Total características: {X_train.shape[1]}")
        
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el archivo {filepath}")
        print("   Descárgalo de: https://huggingface.co/datasets/notadib/NASA-Power-Daily-Weather/")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")