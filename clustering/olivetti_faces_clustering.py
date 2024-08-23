from sklearn.datasets import fetch_olivetti_faces

olivetti = fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target
print(olivetti.DESCR)
print(f"X and y dimensions: {X.shape}, {y.shape}")

