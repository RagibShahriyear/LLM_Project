import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

# Assuming these embeddings are already generated and have the shape (num_items, embedding_dimension)
# For demonstration, I am using random data. Replace these with actual embeddings.
np.random.seed(42)
symptom_embeddings = np.random.rand(3, 768)  # Replace with actual symptom embeddings
medication_embeddings = np.random.rand(3, 768)  # Replace with actual medication embeddings
test_embeddings = np.random.rand(4, 768)  # Replace with actual test embeddings

# Combine all embeddings
all_embeddings = np.concatenate((symptom_embeddings, medication_embeddings, test_embeddings), axis=0)

# Perform PCA to reduce dimensions to 3D
pca = PCA(n_components=3)
all_embeddings_3d = pca.fit_transform(all_embeddings)

# Split the 3D embeddings back to their respective categories
symptom_embeddings_3d = all_embeddings_3d[:3]
medication_embeddings_3d = all_embeddings_3d[3:6]
test_embeddings_3d = all_embeddings_3d[6:]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot symptoms
ax.scatter(symptom_embeddings_3d[:, 0], symptom_embeddings_3d[:, 1], symptom_embeddings_3d[:, 2], c='r', label='Symptoms')

# Plot medications
ax.scatter(medication_embeddings_3d[:, 0], medication_embeddings_3d[:, 1], medication_embeddings_3d[:, 2], c='g', label='Medications')

# Plot tests
ax.scatter(test_embeddings_3d[:, 0], test_embeddings_3d[:, 1], test_embeddings_3d[:, 2], c='b', label='Tests')

# Set plot labels and title
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Embedding Plot of Symptoms, Medications, and Tests')
ax.legend()

# Show plot
plt.show()
