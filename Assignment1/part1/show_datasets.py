import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_samples = np.load('datasets/train_samples_Task4.npy')
test_samples = np.load('datasets/test_samples_Task4.npy')

# Separate the features and labels
train_data = train_samples[:, :2]  # Get the points
train_labels = train_samples[:, 2]  # Get the labels

test_data = test_samples[:, :2]
test_labels = test_samples[:, 2]

# Visualization of the training set
plt.figure(figsize=(8, 6))
plt.scatter(train_data[train_labels == 1][:, 0], train_data[train_labels == 1][:, 1], color='blue', label='Class +1 (Train)')
plt.scatter(train_data[train_labels == -1][:, 0], train_data[train_labels == -1][:, 1], color='red', label='Class -1 (Train)')

# Visualization of the test set
plt.scatter(test_data[test_labels == 1][:, 0], test_data[test_labels == 1][:, 1], edgecolor='blue', facecolor='none', marker='o', label='Class +1 (Test)')
plt.scatter(test_data[test_labels == -1][:, 0], test_data[test_labels == -1][:, 1], edgecolor='red', facecolor='none', marker='o', label='Class -1 (Test)')

# Plot configurations
plt.title("Visualization of Train and Test Points from Two Gaussian Distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig("visualization_train_test_tmp.png")
plt.show()