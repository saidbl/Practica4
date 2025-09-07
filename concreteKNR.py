import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset 
concrete = fetch_ucirepo(id=165)

X = concrete.data.features
y = concrete.data.targets

# Verificar la estructura REAL del dataset
print("=== INFORMACIÓN DEL DATASET ===")
print(f"Nombres de características: {concrete.variables['name'].tolist()}")
print(f"Tipo de X: {type(X)}, Forma: {X.shape}")
print(f"Tipo de y: {type(y)}, Forma: {y.shape}")

# Crear DataFrame con nombres REALES
feature_names = concrete.variables[concrete.variables['role'] == 'Feature']['name']
X = pd.DataFrame(X, columns=feature_names)

# Extraer target correctamente
if hasattr(y, 'values'):
    y = y.values.ravel()
else:
    y = y.iloc[:, 0].values


# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nMuestras entrenamiento: {X_train.shape[0]}")
print(f"Muestras prueba: {X_test.shape[0]}")

# Escalar datos (SOLO si no hay NaN)

if X.isna().sum().sum() == 0:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Crear y entrenar modelo
modelo = KNeighborsRegressor(n_neighbors=5)
modelo.fit(X_train_scaled, y_train)

# Predicciones
y_pred = modelo.predict(X_test_scaled)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== EVALUACIÓN KNeighbors - Concrete ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f} MPa")
print(f"R²: {r2:.4f}")

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="orange", alpha=0.7, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=2, label="Línea Ideal")

# Añadir línea de tendencia
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color="blue", linestyle=":", linewidth=1.5, 
         label=f"Tendencia (R²={r2:.3f})")

plt.xlabel("Valores Reales (Resistencia MPa)", fontsize=12)
plt.ylabel("Predicciones (MPa)", fontsize=12)
plt.title("KNeighborsRegressor - Concrete Compressive Strength", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Añadir cuadro de métricas
textstr = f'MSE = {mse:.2f}\nRMSE = {rmse:.2f} MPa\nR² = {r2:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Información adicional del dataset
print("\n=== INFORMACIÓN ADICIONAL ===")
print("Características del concrete:")
for i, feature in enumerate(feature_names):
    print(f"{i+1}. {feature}")

print(f"\nRango de resistencia: {y.min():.1f} - {y.max():.1f} MPa")
print(f"Resistencia promedio: {y.mean():.2f} MPa")