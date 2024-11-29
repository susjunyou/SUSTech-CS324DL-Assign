from perceptron import Perceptron
from loguru import logger
import numpy as np


def load_data(file_path: str):
    data = np.load(file_path)
    sample = data[:, :-1]
    label = data[:, -1]
    return np.array(sample), np.array(label)

def train(perceptron: Perceptron):
    samples, labels = load_data("datasets/train_samples_0_1_10_1.npy")
    test_samples, test_labels = load_data("datasets/test_samples_0_1_10_1.npy")
    # samples, labels = load_data("datasets/train_samples_Task4.npy")
    # test_samples, test_labels = load_data("datasets/test_samples_Task4.npy")
    perceptron.train(training_inputs=samples, labels=labels, test_inputs=test_samples, test_labels=test_labels)

def test(perceptron: Perceptron):
    samples, labels = load_data("datasets/test_samples_0_1_10_1.npy")
    # samples, labels = load_data("datasets/test_samples_Task4.npy")
    preds = perceptron.forward(samples)
    accuracy = np.sum(preds == labels) / len(samples)
    logger.info(f"accuracy:{accuracy}")



def main():
    perceptron = Perceptron(n_inputs=2, max_epochs=1000, learning_rate=0.01)
    train(perceptron=perceptron)
    test(perceptron=perceptron)
    perceptron.plot_loss()

if __name__ == "__main__":
    main()