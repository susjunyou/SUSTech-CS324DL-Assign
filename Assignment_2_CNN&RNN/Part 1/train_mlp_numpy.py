import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
from loguru import logger
from data_config import load_data
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

def shuffle_data(data, labels):
    """
    Shuffles data and labels.
    
    Args:
        data: np.array of shape (n_samples, n_features)
        labels: np.array of shape (n_samples, n_classes)
    
    Returns:
        shuffled_data: np.array of shape (n_samples, n_features)
        shuffled_labels: np.array of shape (n_samples, n_classes)
    """
    list_data = [(data[i], labels[i]) for i in range(len(data))]
    random.shuffle(list_data)
    shuffled_data = np.array([list_data[i][0] for i in range(len(list_data))])
    shuffled_labels = np.array([list_data[i][1] for i in range(len(list_data))])
    return shuffled_data, shuffled_labels


def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, sgd=False, batch_size=1, type='moon'):
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
    # train_data = np.load('datasets/X_train.npy')
    # train_labels = np.load('datasets/y_train.npy')
    
    # test_data = np.load('datasets/X_test.npy')
    # test_labels = np.load('datasets/y_test.npy')
    train_data, train_labels, test_data, test_labels = load_data(TYPE=type)
    
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
        if sgd:
            
            loss_value_tmp = []
            accuracy_value_tmp = []
            
            # using mini-batch SGD
            train_data, train_label_encoded = shuffle_data(train_data, train_label_encoded)
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                batch_label = train_label_encoded[i:i+batch_size]
                predictions = mlp.forward(batch_data) 
                loss_value_tmp.append(loss.forward(predictions, batch_label) / len(batch_label))
                accuracy_value_tmp.append(accuracy(predictions, batch_label))
                # logger.info(f"Loss: {loss_value}, Accuracy: {accuracy_value}")
                dout = loss.backward(predictions, batch_label)
                mlp.backward(dout)
                mlp.update(learning_rate)
                
                
            loss_value = np.mean(loss_value_tmp)
            accuracy_value = np.mean(accuracy_value_tmp)
            loss_values.append(loss_value)
            accuracy_values.append(accuracy_value)
            
                
                
        else:
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
    
    return loss_values, test_loss_values, accuracy_values, test_accuracy_values
    
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
    parser.add_argument('--sgd', action='store_true', default=False,
                        help='Use stochastic gradient descent')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for mini-batch SGD')
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.sgd, FLAGS.batch_size)

if __name__ == '__main__':
    main()
