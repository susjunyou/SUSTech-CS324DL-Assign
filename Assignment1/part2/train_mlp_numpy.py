import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) * 100

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    train_data = np.load('datasets/X_train.npy')
    train_labels = np.load('datasets/y_train.npy')
    
    test_data = np.load('datasets/X_test.npy')
    test_labels = np.load('datasets/y_test.npy')
    
    encoder = OneHotEncoder(sparse_output=False)
    train_label_encoded = encoder.fit_transform(train_labels.reshape(-1, 1))
    test_label_encoded = encoder.transform(test_labels.reshape(-1, 1))
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here
    mlp = MLP(train_data.shape[1], [int(i) for i in dnn_hidden_units.split(',')], train_label_encoded.shape[1])
    loss = CrossEntropy()
    
    loss_values = []
    test_loss_values = []
    accuracy_values = []
    test_accuracy_values = []
    
    for step in range(max_steps):
        # TODO: Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights
        # random.shuffle(train_data)
        predictions = mlp.forward(train_data) 
        loss_value = loss.forward(predictions, train_label_encoded ) / len(train_label_encoded)
        accuracy_value = accuracy(predictions, train_label_encoded)
        dout = loss.backward(predictions, train_label_encoded)
        mlp.backward(dout)
        mlp.update(learning_rate)
        
        loss_values.append(loss_value)
        accuracy_values.append(accuracy_value)
        
        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            test_predictions = mlp.forward(test_data)
            test_loss = loss.forward(test_predictions, test_label_encoded) / len(test_labels)
            test_accuracy = accuracy(test_predictions, test_label_encoded)
            
            test_loss_values.append(test_loss)
            test_accuracy_values.append(test_accuracy)
            
            print(f"Step: {step}, Loss: {test_loss}, Accuracy: {test_accuracy}, Train Loss: {loss_value}, Train Accuracy: {accuracy_value}")
            
    plot_loss(loss_values, test_loss_values, eval_freq)
    plot_accuracy(accuracy_values, test_accuracy_values, eval_freq)
    
    print("Training complete!")
    
def plot_loss(train_losses, test_losses, eval_freq):
    """
    Plots the loss over time.
    
    Args:
        train_losses: list of training loss values (one per step)
        test_losses: list of test loss values (one per evaluation)
        eval_freq: Frequency at which test loss is evaluated
    """
    plt.clf()
    steps = range(0, len(train_losses))  # All steps for training loss
    eval_steps = range(0, len(train_losses)+1, eval_freq)[:len(train_losses)]  # Only steps where evaluation happened

    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(eval_steps, test_losses, label='Test Loss')  # Use markers to distinguish test loss points
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.png")
    # plt.show()
    
def plot_accuracy(train_accuracies, test_accuracies, eval_freq):
    """
    Plots the accuracy over time.
    
    Args:
        train_accuracies: list of training accuracy values (one per step)
        test_accuracies: list of test accuracy values (one per evaluation)
        eval_freq: Frequency at which test accuracy is evaluated
    """
    plt.clf()
    steps = range(0, len(train_accuracies))  # All steps for training accuracy
    eval_steps = range(0, len(train_accuracies)+1, eval_freq)[:len(train_accuracies)]  # Only steps where evaluation happened

    plt.plot(steps, train_accuracies, label='Train Accuracy')
    plt.plot(eval_steps, test_accuracies, label='Test Accuracy')  # Use markers to distinguish test accuracy points
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy.png")
    # plt.show()
    

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
