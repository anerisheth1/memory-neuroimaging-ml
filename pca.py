from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data

X, y = load_data()

# Choose the number of components you want to keep
n_components = 5

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("PCA-reduced shape:", X_pca.shape)

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained Variance by PCA Components")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(7,6))

# Only use the first two PCA dimensions
pc1 = X_pca[:,0]
pc2 = X_pca[:,1]

plt.scatter(pc1[y==1], pc2[y==1], c='red', label='Memory', s=80)
plt.scatter(pc1[y==0], pc2[y==0], c='blue', label='Non-Memory', s=80)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA Projection of NeuroSynth Maps")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"PC{i+1}: {var:.4f}")
