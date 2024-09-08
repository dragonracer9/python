import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

NR_SAMPLES = 10000
NR_VECS = 100000
VEC_LEN = 8


vec_mat = np.random.rand(NR_VECS, VEC_LEN)
dots = np.zeros(NR_SAMPLES)
angles_rad = np.zeros(NR_SAMPLES)
angles_deg = np.zeros(NR_SAMPLES)

for i in range(NR_SAMPLES):
    idx = np.random.randint(NR_VECS, size=2)
    vecs = vec_mat[idx,:]
    #dot_product = np.dot(vecs[0], vecs[1])
    normalized_vecs = [vec / np.linalg.norm(vec) for vec in vecs]
    dot_product = np.dot(normalized_vecs[0], normalized_vecs[1])
    dots[i] = dot_product
    angles_rad[i] = np.arccos(dot_product / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
    angles_deg[i] = np.degrees(angles_rad[i])
    # print(f"Dot product of vectors {idx[0]} and {idx[1]}: {dot_product}")
    # print(f"Angle between vectors {idx[0]} and {idx[1]}: {angles_rad[i]}")

#print(f"\n\n\n\nNumpy version: {np.__version__}") ## 2.1.1
print(f"-----------------------------")
print(f"Number of vectors: {NR_VECS}")
print(f"Dimensionality of vectors: {VEC_LEN}")
print(f"Number of samples: {NR_SAMPLES}")
print(f"-----------------------------")


print(f"Mean of dot products: {np.mean(dots)}")
print(f"Mean of angles in radians: {np.mean(angles_rad)}")
print(f"Mean of angles in degrees: {np.mean(angles_deg)}\n")

print(f"Standard deviation of dot products: {np.std(dots)}")
print(f"Standard deviation of angles in radians: {np.std(angles_rad)}")
print(f"Standard deviation of angles in degrees: {np.std(angles_deg)}\n")

print(f"Median of dot products: {np.median(dots)}")
print(f"Median of angles in radians: {np.median(angles_rad)}")
print(f"Median of angles in degrees: {np.median(angles_deg)}\n")


counts, bins = np.histogram(angles_deg)
deg = plt.stairs(counts, bins)
plt.savefig('angles_deg.png')

counts, bins = np.histogram(dots)
dot = plt.stairs(counts, bins)
plt.savefig('dots.png')

counts, bins = np.histogram(angles_rad)
rad = plt.stairs(counts, bins)
plt.savefig('angles_rad.png')