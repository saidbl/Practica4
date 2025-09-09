import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Cargar dataset Wine Quality
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features

# Usar nombres reales de características
if hasattr(wine_quality, 'variables'):
    feature_names = wine_quality.variables[wine_quality.variables['role'] == 'Feature']['name']
    X = pd.DataFrame(X, columns=feature_names.tolist())
else:
    X = pd.DataFrame(X)

print("=== Dataset Wine Quality ===")
print(f"Características: {X.shape[1]}, Muestras: {X.shape[0]}")
# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducir con PCA a 2D para visualizar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-Means con 3 clusters (ejemplo)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Graficar clusters
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.6, s=40)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clustering K-Means sobre Wine Quality (PCA reducido a 2D)")
plt.colorbar(label="Cluster")
plt.grid(alpha=0.3)
plt.show()