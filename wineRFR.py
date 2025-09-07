import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

print("=== Dataset Wine Quality ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
print(f"Rango de calidad: {y.min()} - {y.max()}")
print(f"Media de calidad: {y.mean():.2f}")

print("\n=== Primeras filas del dataset ===")
print(X.head())

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nMuestras entrenamiento: {X_train.shape[0]}, Muestras prueba: {X_test.shape[0]}")

# Crear y entrenar modelo RandomForest
modelo = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluación RandomForest - Wine Quality ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} (error en unidades de calidad)")
print(f"MAE: {mae:.4f} (error absoluto promedio)")
print(f"R²: {r2:.4f}")

# Comparar con modelo básico
y_mean = np.full_like(y_test, y_train.mean())
mse_basico = mean_squared_error(y_test, y_mean)
print(f"MSE modelo básico (predecir media): {mse_basico:.4f}")
print(f"Mejora: {((mse_basico - mse)/mse_basico)*100:.1f}%")

# Gráfico: Valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="green", alpha=0.6, s=50, label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2, label="Línea Ideal")

# Añadir línea de tendencia
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color="blue", linestyle=":", linewidth=1.5, 
         label=f"Tendencia (R²={r2:.3f})")

plt.xlabel("Valores Reales (Calidad del Vino)", fontsize=12)
plt.ylabel("Predicciones", fontsize=12)
plt.title("RandomForestRegressor - Wine Quality", fontsize=14)
plt.legend()

# Añadir cuadro de métricas
textstr = f'MSE = {mse:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Importancia de características (valor añadido)
if hasattr(modelo, 'feature_importances_'):
    print("\n=== Importancia de Características ===")
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    # Mostrar las 5 características más importantes
    print("Top 5 características más importantes:")
    for i in range(min(5, len(indices))):
        print(f"{i+1}. {X.columns[indices[i]]}: {importancias[indices[i]]:.4f}")