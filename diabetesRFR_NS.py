import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Cargar dataset (solo X, sin y)
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print("=== Primeras filas del dataset ===")
print(X.head())
# RandomTreesEmbedding (no supervisado)
modelo = RandomTreesEmbedding(n_estimators=100, random_state=42)
X_transform = modelo.fit_transform(X)
print(f"\nDimensiones de la representación generada: {X_transform.shape}")
# Reducir a 2D con PCA para visualizar
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_transform.toarray())
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # probamos con 3 clusters
labels = kmeans.fit_predict(X_transform)
plt.figure(figsize=(6,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="viridis", alpha=0.7, s=20)
plt.scatter(kmeans.cluster_centers_[:,0:1], kmeans.cluster_centers_[:,1:2], 
            c="red", marker="X", s=200, label="Centroides", alpha=0.8)
plt.title("Clustering con KMeans sobre Random Trees Embedding - Diabetes")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.show()
print("\nAsignación de clusters (primeros 10 registros):")
print(labels[:10])
