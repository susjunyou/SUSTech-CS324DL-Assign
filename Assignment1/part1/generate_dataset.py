import numpy as np
np.random.seed(42)


samples_1 = np.random.normal(loc = 0.5, scale = 1, size = (100, 2))
samples_2 = np.random.normal(loc = 0, scale = 1, size = (100, 2))


labels_1 = np.ones((100, 1))
labels_2 = -np.ones((100, 1))


labeled_samples_1 = np.hstack((samples_1, labels_1))
labeled_samples_2 = np.hstack((samples_2, labels_2))

samples = np.vstack((labeled_samples_1, labeled_samples_2))

shuffle = np.random.shuffle(samples)

train_samples = samples[:160]
test_samples = samples[160:]

np.save('datasets/train_samples_Task4.npy', train_samples)
np.save('datasets/test_samples_Task4.npy', test_samples)

# print(samples)