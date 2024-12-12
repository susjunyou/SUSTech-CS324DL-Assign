from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy


def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        outputs = model(batch_inputs)
        optimizer.zero_grad()
        average_loss = criterion(outputs, batch_targets)    

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_norm)
        average_loss.backward()
        optimizer.step()
        losses.update(average_loss.item(), batch_inputs.size(0))
        acc = accuracy(outputs, batch_targets)
        accuracies.update(acc, batch_inputs.size(0))
        # Add more code here ...
        if step % 100 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        outputs = model(batch_inputs)
        average_loss = criterion(outputs, batch_targets)    
        losses.update(average_loss.item(), batch_inputs.size(0))
        acc = accuracy(outputs, batch_targets)
        accuracies.update(acc, batch_inputs.size(0))
        
        if step % 100 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model that we are going to use
    model = LSTM(config.input_length, config.input_dim,
                 config.num_hidden, config.num_classes)
    print(f"input_length: {config.input_length}, input_dim: {config.input_dim}, num_hidden: {config.num_hidden}, num_classes: {config.num_classes}")
    print(model)
    model.to(device)
    # Initialize the dataset and data loader
    dataset = PalindromeDataset(input_length=config.input_length, total_len=config.data_size, one_hot=True)
    # Split dataset into train and validation sets
    train_size, val_size = int(config.portion_train * len(dataset)), len(dataset) - int(config.portion_train * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)


    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(config.max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config)
        scheduler.step()
        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch} | Train Loss {train_loss:.3f} | Train Acc {train_acc:.2f} | Val Loss {val_loss:.3f} | Val Acc {val_acc:.2f}')
        
    plot_accuracy(train_accuracies, val_accuracies)
    plot_loss(train_losses, val_losses)

    print('Done training.')


def plot_loss(train_losses, test_losses):
    """
    Plots the loss over time.
    Args:
        train_losses: list of training loss values (one per step)
        test_losses: list of test loss values (one per evaluation)
    """
    plt.clf()
    steps = range(0, len(train_losses))  

    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, test_losses, label='Test Loss') 
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig("loss.png")
    plt.show()
    
def plot_accuracy(train_accuracies, test_accuracies):
    """
    Plots the accuracy over time.
    Args:
        train_accuracies: list of training accuracy values (one per step)
        test_accuracies: list of test accuracy values (one per evaluation)
    """
    plt.clf()
    steps = range(0, len(train_accuracies))  
    plt.plot(steps, train_accuracies, label='Train Accuracy')
    plt.plot(steps, test_accuracies, label='Test Accuracy')  
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig("accuracy.png")
    plt.show()

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
