from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
# EVAL_FREQ_DEFAULT = 500
EVAL_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = "/home/zhe/cs224n/DeepLearning/Assignment_2_CNN&RNN/Part 1/data"

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
    accuracy = torch.mean((predicted_labels == targets).float()) * 100
    return accuracy

def train(learning_rate, max_steps, batch_size, eval_freq, trainSet, testSet):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # print all the hyperparameters
    print(f'Hyperparameters: learning_rate={learning_rate}, max_steps={max_steps}, eval_freq={eval_freq}, batch_size={batch_size}')
    
    # Data loading
    trainLoder = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,persistent_workers=True)
    testLoder = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,persistent_workers=True)
    
    n_channels = trainSet.data.shape[3]
    n_classes = len(trainSet.classes)
    model = CNN(n_channels, n_classes).to(device)  # Move model to device
    
    if torch.cuda.device_count() > 1:
        print("use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
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
            train_acc += accuracy(y_pred, y).item()
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
                    test_acc += accuracy(y_pred, y).item()
                test_loss /= len(testLoder)
                test_acc /= len(testLoder)
                
                test_losses.append(test_loss)
                test_acces.append(test_acc)
                print(f'Step {step}: Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}')
                
    
    np.save('data/train_losses.npy', np.array(train_losses))
    np.save('data/train_acces.npy', np.array(train_acces))
    np.save('data/test_losses.npy', np.array(test_losses))
    np.save('data/test_acces.npy', np.array(test_acces))
    
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

def main(leanring_rate, max_steps, batch_size, eval_freq, data_dir):
    """
    Main function
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainSet = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    testSet = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    train(learning_rate=leanring_rate, max_steps=max_steps, batch_size=batch_size, eval_freq=eval_freq, trainSet=trainSet, testSet=testSet)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main(leanring_rate=FLAGS.learning_rate, max_steps=FLAGS.max_steps, batch_size=FLAGS.batch_size, eval_freq=FLAGS.eval_freq, data_dir=FLAGS.data_dir)