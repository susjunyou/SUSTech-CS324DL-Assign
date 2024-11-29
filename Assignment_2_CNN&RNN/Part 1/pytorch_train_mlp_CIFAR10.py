from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib.pyplot as plt
import torch.optim.sgd
from pytorch_mlp_CIFAR10 import MLP
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from loguru import logger

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '1024,512,256,256,128'
LEARNING_RATE_DEFAULT = 1e-3
MAX_EPOCHS_DEFAULT = 50
EVAL_FREQ_DEFAULT = 1
BATCH_SIZE_DEFAULT = 256

def accuracy(predictions, targets):
    predicted_labels = torch.argmax(predictions, axis=1)
    accuracy = torch.mean((predicted_labels == targets).float()) * 100
    return accuracy.item()

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size, trainSet, testSet):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # print all the hyperparameters
    print(f'Hyperparameters: dnn_hidden_units={dnn_hidden_units}, learning_rate={learning_rate}, max_steps={max_steps}, eval_freq={eval_freq}, batch_size={batch_size}')



    # Data loading
    trainLoder = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=8)
    testLoder = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=8)
    
    n_inputs = trainSet.data.shape[1] * trainSet.data.shape[2] * trainSet.data.shape[3]
    n_classes = len(trainSet.classes)
    n_hidden = list(map(int, dnn_hidden_units.split(',')))

    model = MLP(n_inputs, n_hidden, n_classes).to(device)  # Move model to device

    if torch.cuda.device_count() > 1:
        print("use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    # Loss and optimizer
    loss = nn.CrossEntropyLoss().to(device)  # Move loss to device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f'Starting Training Model{model}')

    train_losses, train_acces, test_losses, test_acces = [], [], [], []

    for step in range(max_steps):
        model.train()
        train_loss = 0
        train_acc = 0
        for x, y in trainLoder:
            # Move data to device
            x, y = x.to(device), y.to(device)
            # Forward pass
            y_pred = model.forward(x)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_acc += accuracy(y_pred, y)
        train_loss /= len(trainLoder)
        train_acc /= len(trainLoder)
        
        train_losses.append(train_loss)
        train_acces.append(train_acc)

        if step % eval_freq == 0:
            model.eval()

            test_loss = 0
            test_acc = 0
            
            # Calculate Accuracy and Loss on Training set
            with torch.no_grad():

                # Calculate Accuracy and Loss on Test set
                for x, y in testLoder:
                    x, y = x.to(device), y.to(device)
                    y_pred = model.forward(x)
                    l = loss(y_pred, y)
                    test_loss += l.item()
                    test_acc += accuracy(y_pred, y)
                test_loss /= len(testLoder)
                test_acc /= len(testLoder)
                test_losses.append(test_loss)
                test_acces.append(test_acc)
            print(f'Step {step}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}')
            
    plot_loss(train_losses, test_losses, eval_freq)
    plot_accuracy(train_acces, test_acces, eval_freq)

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
    eval_steps = range(0, len(train_losses)+1, eval_freq)[:len(train_losses)]  

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
    eval_steps = range(0, len(train_accuracies)+1, eval_freq)[:len(train_accuracies)]  
    plt.plot(steps, train_accuracies, label='Train Accuracy')
    plt.plot(eval_steps, test_accuracies, label='Test Accuracy')  
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig("accuracy.png")
    plt.show()



def main():
    
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
    FLAGS, unparsed = parser.parse_known_args()
    
    # Load CIFAR-10 data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testSet = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # Train the model
    train(
        dnn_hidden_units=FLAGS.dnn_hidden_units,
        learning_rate=FLAGS.learning_rate,
        max_steps=FLAGS.max_steps,
        eval_freq=FLAGS.eval_freq,
        batch_size=FLAGS.batch_size,
        trainSet=trainSet,
        testSet=testSet
    )

if __name__ == '__main__':

    main()

