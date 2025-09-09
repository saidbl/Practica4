import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
# 1. Cargar dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
# 2. Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 3. PCA (modelo lineal de reducción de dimensionalidad)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("=== Varianza explicada por PCA ===")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"Componente {i+1}: {var:.4f}")
print(f"Varianza total explicada (2 componentes): {np.sum(pca.explained_variance_ratio_):.4f}")
# 4. Clustering con KMeans sobre el espacio PCA
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
# 5. Graficar PCA + Clusters
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            c="red", marker="X", s=200, label="Centroides")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clustering (KMeans) sobre PCA lineal - Diabetes")
plt.legend()
plt.show()
