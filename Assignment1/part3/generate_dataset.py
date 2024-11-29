from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

# Create a dataset of 1000 points
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save('datasets/X_train.npy', X_train)
np.save('datasets/X_test.npy', X_test)
np.save('datasets/y_train.npy', y_train)
np.save('datasets/y_test.npy', y_test)

