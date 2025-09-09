import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors  
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1) Cargar dataset (SOLO características - no supervisado)
concrete = fetch_ucirepo(id=165)
X = concrete.data.features

# Verificar la estructura REAL del dataset
print("=== INFORMACIÓN DEL DATASET PARA CLUSTERING ===")
print(f"Nombres de características: {concrete.variables['name'].tolist()}")
print(f"Tipo de X: {type(X)}, Forma: {X.shape}")

# Crear DataFrame con nombres REALES
feature_names = concrete.variables[concrete.variables['role'] == 'Feature']['name']
X = pd.DataFrame(X, columns=feature_names)

print(f"\nMuestras totales: {X.shape[0]}")
print("Columnas:", ", ".join(X.columns))

# 2) Procesamiento para clustering
if X.isna().sum().sum() > 0:
    X_imp = X.fillna(X.median(numeric_only=True))
    print(f"Valores NaN imputados: {X.isna().sum().sum()}")
else:
    X_imp = X.copy()
    print("No hay valores NaN en el dataset")

# Escalado estándar (CRUCIAL para clustering basado en distancias)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 3) CLUSTERING NO SUPERVISADO con NearestNeighbors + DBSCAN approach

# Primero, análisis de distancias para determinar parámetros
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)

# Calcular distancias promedio a los k-vecinos más cercanos
k_distances = np.sort(distances[:, -1])[::-1]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_distances)
plt.title('Distancias a 5° vecino más cercano')
plt.xlabel('Puntos ordenados por distancia')
plt.ylabel('Distancia')
plt.grid(True, alpha=0.3)

# 4) Clustering basado en densidad usando KNeighbors

from sklearn.cluster import DBSCAN

# Determinar epsilon automáticamente (percentil 95 de distancias)
epsilon = np.percentile(distances[:, -1], 95)
dbscan = DBSCAN(eps=epsilon, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Si DBSCAN encuentra muchos outliers (-1), usar KMeans alternativo
if np.sum(labels == -1) > len(X_scaled) * 0.3:
    print("Demasiados outliers con DBSCAN, usando KMeans...")
    from sklearn.cluster import KMeans
    
    # Selección automática de k con silhouette
    K_range = range(2, 11)
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels_k = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels_k))
    
    best_k = K_range[np.argmax(sil_scores)]
    kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    n_clusters = best_k
else:
    n_clusters = len(np.unique(labels[labels != -1]))
    print(f"Clusters encontrados por DBSCAN: {n_clusters}")

print(f"\nNúmero de clusters: {n_clusters}")

# 5) Evaluación de la calidad del clustering
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score:.3f}")

# 6) Análisis de los clusters
df_clusters = X_imp.copy()
df_clusters['cluster'] = labels

# Estadísticas por cluster
print("\n=== ESTADÍSTICAS POR CLUSTER ===")
cluster_stats = df_clusters.groupby('cluster').agg(['mean', 'std', 'count'])
print(cluster_stats.round(2))

# Distribución de clusters
cluster_counts = df_clusters['cluster'].value_counts().sort_index()
print(f"\nDistribución de puntos por cluster:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} puntos ({count/len(X_scaled)*100:.1f}%)")

# 7) Visualización con PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

plt.subplot(1, 2, 2)
unique_clusters = np.unique(labels)
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

for i, cluster in enumerate(unique_clusters):
    if cluster == -1:
        color = 'gray'
        label = 'Outliers'
        alpha = 0.6
        size = 20
    else:
        color = colors[i]
        label = f'Cluster {cluster}'
        alpha = 0.7
        size = 40
    
    mask = labels == cluster
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                c=[color], alpha=alpha, s=size, label=label)

plt.title(f'Clustering - PCA (Silhouette: {sil_score:.3f})')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

