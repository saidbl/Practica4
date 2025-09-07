import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Cargar dataset Concrete Compressive Strength (id=165)
concrete = fetch_ucirepo(id=165)
X = concrete.data.features
y = concrete.data.targets

if hasattr(y, 'values'):
    y = y.values.ravel()  # Convertir a array 1D
else:
    y = y.iloc[:, 0].values  # Si es DataFrame, tomar primera columna

# Usar nombres reales de características si están disponibles
if hasattr(concrete, 'variables'):
    feature_names = concrete.variables[concrete.variables['role'] == 'Feature']['name']
    X = pd.DataFrame(X, columns=feature_names.tolist())
else:
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

print("=== Dataset Concrete Compressive Strength ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
print(f"Rango de resistencia: {y.min():.1f} - {y.max():.1f} MPa")
print(f"Resistencia media: {y.mean():.2f} MPa")
print(f"Desviación estándar: {y.std():.2f} MPa")

print("\n=== Primeras filas del dataset ===")
print(X.head())

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nMuestras entrenamiento: {X_train.shape[0]}, Muestras prueba: {X_test.shape[0]}")

# Crear y entrenar modelo LinearRegression
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Mostrar coeficientes del modelo
print("\n=== Coeficientes del Modelo (8 más importantes) ===")
coeficientes = pd.DataFrame({
    'Característica': X.columns,
    'Coeficiente': modelo.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)

print(coeficientes.head(8))
print(f"Intercepto: {modelo.intercept_:.4f}")

# Predicciones
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluación LinearRegression - Concrete ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} MPa (error en unidades de resistencia)")
print(f"MAE: {mae:.4f} MPa (error absoluto promedio)")
print(f"R²: {r2:.4f}")

# Comparar con modelo básico (predecir la media)
y_mean = np.full_like(y_test, y_train.mean())
mse_basico = mean_squared_error(y_test, y_mean)
r2_basico = r2_score(y_test, y_mean)
print(f"\nComparación con modelo básico:")
print(f"MSE (predecir media): {mse_basico:.4f}")
print(f"R² (predecir media): {r2_basico:.4f}")
print(f"Mejora en MSE: {((mse_basico - mse)/mse_basico)*100:.1f}%")

# Gráfico: Valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, s=50, label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2, label="Línea Ideal")

# Añadir línea de tendencia
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color="green", linestyle=":", linewidth=1.5, 
         label=f"Tendencia (R²={r2:.3f})")

plt.xlabel("Valores Reales (Resistencia a Compresión, MPa)", fontsize=12)
plt.ylabel("Predicciones (MPa)", fontsize=12)
plt.title("LinearRegression - Concrete Compressive Strength", fontsize=14)
plt.legend()

# Añadir cuadro de métricas
textstr = f'MSE = {mse:.2f}\nRMSE = {rmse:.2f} MPa\nR² = {r2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()