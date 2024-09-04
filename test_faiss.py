import numpy as np
import faiss


d = 64  
nb = 1000 
nq = 10 

np.random.seed(1234)  # for reproducibility

# Generate random database and query vectors
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Normalize the vectors to unit length
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

# Create the FAISS index
index = faiss.IndexFlatL2(d)  # L2 distance index

# Add the database vectors to the index
index.add(xb)

# Check the number of vectors in the index
print(f"Number of vectors in the index: {index.ntotal}")

# Perform a search
k = 5  # number of nearest neighbors to retrieve
D, I = index.search(xq, k)  # distances and indices of the k nearest neighbors

# Print the results
print("Indices of nearest neighbors:\n", I)
print("Distances to nearest neighbors:\n", D)
