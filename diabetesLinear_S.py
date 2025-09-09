import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

print("=== Primeras filas del dataset ===")
print(X.head())
print("\n=== Valores objetivo (target) ===")
print(y.head())

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nNúmero de muestras de entrenamiento: {X_train.shape[0]}")
print(f"Número de muestras de prueba: {X_test.shape[0]}")

# Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("\n=== Coeficientes del modelo ===")
for nombre, coef in zip(X.columns, modelo.coef_):
    print(f"{nombre}: {coef:.4f}")

print(f"Intercepto: {modelo.intercept_:.4f}")

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluación del modelo ===")
print(f"MSE: {mse:.4f}")
print(f"R² : {r2:.4f}")

# Gráfico: Valores reales vs predichos
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.7, label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Línea Ideal")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("LinearRegression - Diabetes")
plt.legend()
plt.show()