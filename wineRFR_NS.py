import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# 1. Cargar dataset Wine Quality
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
# Nombres de características
if hasattr(wine_quality, 'variables'):
    feature_names = wine_quality.variables[wine_quality.variables['role'] == 'Feature']['name']
    X = pd.DataFrame(X, columns=feature_names.tolist())
else:
    X = pd.DataFrame(X)
print("=== Dataset Wine Quality ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
print("\nPrimeras 5 filas del dataset:")
print(X.head())
# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 2. Crear etiquetas sintéticas: datos reales (1) + datos permutados (0)
np.random.seed(42)
X_perm = np.random.permutation(X_scaled)
X_mix = np.vstack((X_scaled, X_perm))
y_mix = np.hstack((np.ones(X_scaled.shape[0]), np.zeros(X_perm.shape[0])))
# 3. Entrenar Random Forest para aprender la estructura de los datos
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_mix, y_mix)
# 4. Construir matriz de proximidad (medida de similitud entre muestras)
leaves = rf.apply(X_scaled)  # hoja de cada muestra en cada árbol
n_samples = X_scaled.shape[0]
proximity = np.zeros((n_samples, n_samples))
for tree in range(leaves.shape[1]):
    leaf_ids = leaves[:, tree]
    for leaf in np.unique(leaf_ids):
        idx = np.where(leaf_ids == leaf)[0]
        for i in idx:
            proximity[i, idx] += 1
# Normalizar (frecuencia de co-ocurrencia en hojas)
proximity /= rf.n_estimators
# 5. Reducción dimensional con PCA para visualización
pca = PCA(n_components=2)
X_proj = pca.fit_transform(proximity)
# 6. Determinar número óptimo de clusters con método del codo
inertias = []
silhouette_scores = []
k_range = range(2, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_proj)
    inertias.append(kmeans.inertia_)
    
    if k > 1:  # Silhouette score necesita al menos 2 clusters
        score = silhouette_score(X_proj, kmeans.labels_)
        silhouette_scores.append(score)
# 7. Aplicar K-means con el número óptimo de clusters (elegimos 3 basado en el codo)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_proj)
# 8. Visualización de los clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroides")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clustering con Matriz de Proximidad de Random Forest - Wine Quality")
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)
plt.show()
# 13. Evaluación del clustering
silhouette_avg = silhouette_score(X_proj, labels)
print(f"\nPuntuación Silhouette: {silhouette_avg:.3f}")
