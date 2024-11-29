from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

import torch.optim.sgd
from pytorch_mlp import MLP
from data_config import load_data   
import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger
import matplotlib.pyplot as plt
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
BATCH_SIZE_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predicted_labels = torch.argmax(predictions, axis=1)
    target_labels =torch.argmax(targets, axis=1)
    accuracy = torch.mean((predicted_labels == target_labels).float()) * 100
    return accuracy.item()



def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size, type='moon'):
    """
    Performs training and evaluation of MLP model.
    Args:
        dnn_hidden_units: str, comma-separated list of number of units in each hidden layer
        learning_rate: float, learning rate for SGD
        max_steps: int, number of epochs to run trainer
        eval_freq: int, frequency of evaluation on the test set
        batch_size: int, batch size to train
    """
    # Load data
    x_train, y_train, x_test, y_test = load_data(TYPE=type)
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_encoded, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_inputs = x_train.shape[1]
    n_classes = y_train_encoded.shape[1]
    n_hidden = list(map(int, dnn_hidden_units.split(',')))

    model = MLP(n_inputs, n_hidden, n_classes)

    # Loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print(f'Starting Training Model {model}')

    train_loss, train_acc, test_loss, test_acc = [], [], [], []

    for step in range(max_steps):
        for x, y in dataloader:
            # Forward pass
            y_pred = model.forward(x)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        y_pred = model.forward(torch.tensor(x_train, dtype=torch.float32))
        l = loss(y_pred, torch.tensor(y_train_encoded, dtype=torch.float32))
        train_loss.append(l.item())
        train_acc.append(accuracy(y_pred, torch.tensor(y_train_encoded, dtype=torch.float32)))

        if step % eval_freq == 0:
            y_pred_test = model.forward(torch.tensor(x_test, dtype=torch.float32))
            l_test = loss(y_pred_test, torch.tensor(y_test_encoded, dtype=torch.float32))
            test_loss.append(l_test.item())
            test_acc.append(accuracy(y_pred_test, torch.tensor(y_test_encoded, dtype=torch.float32)))
            print(f'Step {step}, Train Loss: {l}, Train Accuracy: {train_acc[-1]}, Test Loss: {l_test}, Test Accuracy: {test_acc[-1]}')

    plot_loss(train_loss, test_loss, eval_freq)
    plot_accuracy(train_acc, test_acc, eval_freq)

    print('Finished Training')
    
    
def plot_loss(train_losses, test_losses, eval_freq):
    """
    Plots the loss over time.
    
    Args:
        train_losses: list of training loss values (one per step)
        test_losses: list of test loss values (one per evaluation)
        eval_freq: Frequency at which test loss is evaluated
    """
    plt.clf()
    steps = range(0, len(train_losses))  
    eval_steps = [x*eval_freq for x in range(len(test_losses))]

    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(eval_steps, test_losses, label='Test Loss') 
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig("loss.png")
    plt.show()
    
def plot_accuracy(train_accuracies, test_accuracies, eval_freq):
    """
    Plots the accuracy over time.
    
    Args:
        train_accuracies: list of training accuracy values (one per step)
        test_accuracies: list of test accuracy values (one per evaluation)
        eval_freq: Frequency at which test accuracy is evaluated
    """
    plt.clf()
    steps = range(0, len(train_accuracies))  
    eval_steps = [x*eval_freq for x in range(len(test_accuracies))] 
    plt.plot(steps, train_accuracies, label='Train Accuracy')
    plt.plot(eval_steps, test_accuracies, label='Test Accuracy')  
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig("accuracy.png")
    plt.show()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to train')
    args = parser.parse_args()
    
    train(args.dnn_hidden_units, args.learning_rate, args.max_steps, args.eval_freq, args.batch_size)

if __name__ == '__main__':
    main()