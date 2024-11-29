import numpy as np


def load_data(TYPE: str = "moon"):
    """
    Load dataset.
    Returns:
        X_train: 2D float array of size [number_of_training_samples, input_dim]
        Y_train: 2D int array of size [number_of_training_samples, n_classes]
        X_test: 2D float array of size [number_of_test_samples, input_dim]
        Y_test: 2D int array of size [number_of_test_samples, n_classes]
    """
    
    assert TYPE in ["moon", "blob", "circle"], "Invalid dataset type"
    
    # Load dataset
    X_train = np.load("data/X_train_" + TYPE + ".npy")
    Y_train = np.load("data/y_train_" + TYPE + ".npy")
    X_test = np.load("data/X_test_" + TYPE + ".npy")
    Y_test = np.load("data/y_test_" + TYPE + ".npy")
    return X_train, Y_train, X_test, Y_test