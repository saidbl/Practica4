import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Cargar dataset Concrete Compressive Strength (id=165) - SOLO características
concrete = fetch_ucirepo(id=165)
X = concrete.data.features

# Usar nombres reales de características si están disponibles
if hasattr(concrete, 'variables'):
    feature_names = concrete.variables[concrete.variables['role'] == 'Feature']['name']
    X = pd.DataFrame(X, columns=feature_names.tolist())
else:
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

print("=== Dataset Concrete - Análisis NO SUPERVISADO ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
print("\nCaracterísticas:", ", ".join(X.columns))

print("\n=== Primeras filas del dataset ===")
print(X.head())

# 1) PREPROCESAMIENTO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) - Modelo Lineal
pca = PCA(random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("\n=== ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) ===")
print(f"Varianza explicada por cada componente:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.4f} ({var*100:.1f}%)")

print(f"\nVarianza acumulada:")
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
for i, var in enumerate(cumulative_var):
    print(f"PC1-{i+1}: {var:.4f} ({var*100:.1f}%)")

# 3) DETERMINAR NÚMERO ÓPTIMO DE COMPONENTES¡
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% varianza')
plt.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% varianza')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada Explicada')
plt.title('Varianza Explicada por Componentes')
plt.legend()
plt.grid(True, alpha=0.3)

# 4) CLUSTERING EN ESPACIO REDUCIDO (K-Means)
K_range = range(2, 11)
sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = K_range[np.argmax(sil_scores)]
best_sil = max(sil_scores)

print(f"\n=== CLUSTERING NO SUPERVISADO ===")
print(f"Mejor número de clusters (silhouette): {best_k}")
print(f"Silhouette Score: {best_sil:.3f}")

kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# 5) VISUALIZACIÓN 2D CON PCA
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels, cmap='viridis', 
                     alpha=0.7, s=50, edgecolor='k', linewidth=0.5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title(f'Clustering en 2D PCA (k={best_k}, Silhouette={best_sil:.3f})')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()





