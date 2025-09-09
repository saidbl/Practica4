import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Cargar dataset Wine Quality (id=186)
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

if hasattr(y, 'values'):
    y = y.values.ravel()  # Convertir a array 1D
else:
    y = y.iloc[:, 0].values  # Si es DataFrame, tomar primera columna

# Usar nombres reales de características si están disponibles
if hasattr(wine_quality, 'variables'):
    feature_names = wine_quality.variables[wine_quality.variables['role'] == 'Feature']['name']
    X = pd.DataFrame(X, columns=feature_names.tolist())
else:
    X = pd.DataFrame(X)

print("=== Dataset Wine Quality ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
print(f"Rango de calidad: {y.min()} - {y.max()}")
print(f"Media de calidad: {y.mean():.2f}")

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nMuestras entrenamiento: {X_train.shape[0]}, Muestras prueba: {X_test.shape[0]}")

# Escalar datos (IMPORTANTE para KNN)
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
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluación KNeighbors - Wine Quality ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} (error en unidades de calidad)")
print(f"MAE: {mae:.4f} (error absoluto promedio)")
print(f"R²: {r2:.4f}")

# Comparar con modelo básico (predecir la media)
y_mean = np.full_like(y_test, y_train.mean())
mse_basico = mean_squared_error(y_test, y_mean)
print(f"MSE modelo básico (predecir media): {mse_basico:.4f}")
print(f"Mejora: {((mse_basico - mse)/mse_basico)*100:.1f}%")

# Gráfico: Valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="orange", alpha=0.7, s=50, label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2, label="Línea Ideal")

# Añadir línea de tendencia
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color="green", linestyle=":", linewidth=1.5, 
         label=f"Tendencia (R²={r2:.3f})")

plt.xlabel("Valores Reales (Calidad del Vino)", fontsize=12)
plt.ylabel("Predicciones", fontsize=12)
plt.title("KNeighborsRegressor - Wine Quality Dataset", fontsize=14)
plt.legend()

# Añadir cuadro de métricas
textstr = f'MSE = {mse:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Optimización opcional: encontrar mejor k
print("\n=== Optimización: Buscando mejor valor de k ===")
k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

# Encontrar k óptimo
best_k = k_values[np.argmax(test_scores)]
print(f"Mejor k: {best_k} (R² = {max(test_scores):.4f})")