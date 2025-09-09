import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RANDOM_STATE = 42
concrete = fetch_ucirepo(id=165)
X = pd.DataFrame(
    concrete.data.features,
    columns=concrete.variables[concrete.variables['role'] == 'Feature']['name']
)
print("=== ANÁLISIS NO SUPERVISADO: CLUSTERING CON PROXIMIDADES RF ===")
print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")

# 1) PREPROCESAMIENTO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) Unsupervised Random Forest (URF)

def make_permuted_copy(X_np, rng):
    X_perm = X_np.copy()
    for j in range(X_perm.shape[1]):
        rng.shuffle(X_perm[:, j])
    return X_perm

rng = np.random.default_rng(RANDOM_STATE)
X_perm = make_permuted_copy(X_scaled.copy(), rng)

X_urf = np.vstack([X_scaled, X_perm])
y_urf = np.hstack([np.ones(X_scaled.shape[0], dtype=int),
                   np.zeros(X_perm.shape[0], dtype=int)])

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,   # solo como chequeo interno de discriminabilidad real vs sintético
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_urf, y_urf)
print(f"OOB score (real vs sintético): {rf.oob_score_:.3f}")

# 3) MATRIZ DE PROXIMIDADES (solo entre muestras REALES)
n_real = X_scaled.shape[0]
proximity = np.zeros((n_real, n_real), dtype=np.float32)
for tree in rf.estimators_:
    leaf_ids = tree.apply(X_urf)        
    leaf_real = leaf_ids[:n_real]      
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, leaf in enumerate(leaf_real):
        buckets[leaf].append(idx)
    for inds in buckets.values():
        inds = np.asarray(inds, dtype=int)
        proximity[np.ix_(inds, inds)] += 1
proximity /= len(rf.estimators_)
np.fill_diagonal(proximity, 1.0)

# 4) DISTANCIAS + CLUSTERING (k de 2 a 10) con métrica precomputada
distance = 1.0 - proximity
K_RANGE = range(2, 11)

best_k, best_sil, best_labels = None, -1.0, None

for k in K_RANGE:
    agg = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",   
        linkage="average"
    )
    labels = agg.fit_predict(distance)
    sil = silhouette_score(distance, labels, metric="precomputed")
    if sil > best_sil:
        best_k, best_sil, best_labels = k, sil, labels

print(f"Mejor número de clusters: {best_k}")
print(f"Silhouette (distancia precomputada): {best_sil:.3f}")

# 5) VISUALIZACIÓN en 2D con PCA (solo para ver separaciones)
pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
X_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=best_labels, alpha=0.8)
plt.title(f'Clustering no supervisado (URF + proximidades)  |  k={best_k}')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# 6) "Importancia" de características (desde el bosque real vs sintético)
importances = rf.feature_importances_
importance_df = (
    pd.DataFrame({"feature": X.columns, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\nImportancia de características (estructura vs permutación):")
print(importance_df)
# 7) Pequeño resumen por cluster (medianas y tamaño)
summary = (
    pd.concat([pd.DataFrame(X_scaled, columns=X.columns), pd.Series(best_labels, name="cluster")], axis=1)
    .groupby("cluster")
    .agg(["median", "mean", "std", "count"])
)
print("\nResumen por cluster (sobre datos estandarizados):")
print(summary)
