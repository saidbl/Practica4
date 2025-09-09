import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Cargar dataset Diabetes 
diabetes = load_diabetes() 
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

print(f"\n=== INFORMACIÓN DEL DATASET DIABETES ===") 
print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}") 
print("=== Primeras filas del dataset ===") 
print(X.head())

# Escalar datos 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# 1. MODELO NEARESTNEIGHBORS PARA ENCONTRAR VECINOS CERCANOS
print("\n=== NEAREST NEIGHBORS - ANÁLISIS NO SUPERVISADO ===")

nn_model = NearestNeighbors(n_neighbors=6, algorithm='auto')  # n_neighbors=6 para incluir el punto mismo
nn_model.fit(X_scaled)

distances, indices = nn_model.kneighbors(X_scaled[:5]) 

print("\nVecinos más cercanos para las primeras 5 muestras:")
for i in range(5):
    print(f"Muestra {i}:")
    print(f"  - Vecinos: {indices[i][1:].tolist()}")
    distancias_formateadas = [f"{d:.4f}" for d in distances[i][1:]]
    print(f"  - Distancias: {distancias_formateadas}")

# 2. DETECCIÓN DE OUTLIERS BASADA EN DISTANCIAS
print("\n=== DETECCIÓN DE OUTLIERS ===")

distances_all, _ = nn_model.kneighbors(X_scaled)
avg_distances = distances_all[:, 1:].mean(axis=1)  
outlier_threshold = np.percentile(avg_distances, 95)  # Percentil 95
outliers = np.where(avg_distances > outlier_threshold)[0]

print(f"Umbral de outlier: {outlier_threshold:.4f}")
print(f"Número de outliers detectados: {len(outliers)}")
print(f"Índices de outliers: {outliers[:10].tolist()}")  # Mostrar primeros 10

# 3. CLUSTERING CON K-MEANS (COMPARACIÓN)
print("\n=== CLUSTERING CON K-MEANS ===")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print(f"Asignación de clusters para primeras 10 muestras: {clusters[:10].tolist()}")
print(f"Tamaño de clusters: {np.bincount(clusters).tolist()}")

# 4. REDUCCIÓN DE DIMENSIONALIDAD CON PCA PARA VISUALIZACIÓN
print("\n=== REDUCCIÓN DE DIMENSIONALIDAD (PCA) ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Varianza explicada por componentes: {pca.explained_variance_ratio_.tolist()}")
print(f"Varianza total explicada: {pca.explained_variance_ratio_.sum():.3f}")
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=50)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters en Espacio Reducido (PCA)")
plt.colorbar(label="Cluster")
plt.show()


