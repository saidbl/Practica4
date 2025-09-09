import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Cargar dataset Wine Quality (id=186)
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features

# Usar nombres reales de características si están disponibles
if hasattr(wine_quality, 'variables'):
    feature_names = wine_quality.variables[wine_quality.variables['role'] == 'Feature']['name']
    X = pd.DataFrame(X, columns=feature_names.tolist())
else:
    X = pd.DataFrame(X)

print("=== Dataset Wine Quality ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=5, metric="euclidean")
clusters = dbscan.fit_predict(X_scaled)

# Reducción PCA para visualizar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualización
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clustering basado en Vecinos (DBSCAN) - Wine Quality")
plt.colorbar(label="Cluster")
plt.grid(alpha=0.3)
plt.show()