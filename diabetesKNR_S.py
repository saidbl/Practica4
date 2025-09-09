import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset Diabetes
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

print(f"\n=== INFORMACIÓN DEL DATASET DIABETES ===")
print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
print(f"Target range: {y.min()} - {y.max()}")
print(f"Target mean: {y.mean():.2f} ± {y.std():.2f}") 

print("=== Primeras filas del dataset ===")
print(X.head())
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nNúmero de muestras de entrenamiento: {X_train.shape[0]}")
print(f"Número de muestras de prueba: {X_test.shape[0]}")

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar modelo KNeighbors
modelo = KNeighborsRegressor(n_neighbors=5)
modelo.fit(X_train_scaled, y_train)

# Predicciones
y_pred = modelo.predict(X_test_scaled)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluación KNeighbors - Diabetes ===")
print(f"MSE: {mse:.4f}")
print(f"R² : {r2:.4f}")

# Gráfico: Valores reales vs predichos
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, color="orange", alpha=0.7, label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Línea Ideal")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("KNeighborsRegressor - Diabetes")
plt.legend()
plt.show()
