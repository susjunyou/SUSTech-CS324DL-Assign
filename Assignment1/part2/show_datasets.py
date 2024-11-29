import numpy as np
import matplotlib.pyplot as plt

# 加载保存的数据集
X_train = np.load('datasets/X_train.npy')
X_test = np.load('datasets/X_test.npy')
y_train = np.load('datasets/y_train.npy')
y_test = np.load('datasets/y_test.npy')

# 可视化训练数据
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train[:, 0] == 0, 0], X_train[y_train[:, 0] == 0, 1], color='red', label='Class 0')
plt.scatter(X_train[y_train[:, 0] == 1, 0], X_train[y_train[:, 0] == 1, 1], color='blue', label='Class 1')
plt.title('Visualization of Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("visualization_train.png")

plt.clf()

# 可视化测试数据
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test[:, 0] == 0, 0], X_test[y_test[:, 0] == 0, 1], color='red', label='Class 0')
plt.scatter(X_test[y_test[:, 0] == 1, 0], X_test[y_test[:, 0] == 1, 1], color='blue', label='Class 1')
plt.title('Visualization of Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.savefig("visualization_test.png")