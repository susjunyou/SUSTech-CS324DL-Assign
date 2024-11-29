import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=2e-5):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(n_inputs+1)  # Fill in: Initialize weights with zeros 
        self.losses = []
        self.test_losses = []
        pass
        
    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        # (n, 2) * (2, 1) = (n, 1) 
        # add bias term to the input_vec
        # logger.debug(f"input_vec: {input_vec}")
        # logger.debug(f"weights: {self.weights}")
        result = np.sign(np.matmul(input_vec, self.weights[1:]) + self.weights[0]) 
        result = np.array([i if i != 0 else 1 for i in result])
        return result
        
    def train(self, training_inputs, labels, test_inputs, test_labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        
        # we need max_epochs to train our model
        for _ in range(self.max_epochs):
            """
            What we should do in one epoch ? 
            you are required to write code for 
            1.do forward pass
            2.calculate the error
            3.compute parameters' gradient 
            4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
            please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            preds = self.forward(training_inputs)
            error_position = np.where(labels * preds < 0)
            error_input = training_inputs[error_position]
            error_label = labels[error_position]
            error_pred = preds[error_position]
            # calculate the loss
            loss = np.sum( -error_label * error_pred) / len(training_inputs)
            self.losses.append(loss)
            logger.info(f"loss: {loss}")
            # calculate the gradient
            gradient = np.sum(-error_label[:, np.newaxis] * error_input, axis=0)
            # update the weights
            self.weights[1:] -= self.learning_rate * gradient
            self.weights[0] -= self.learning_rate * np.sum(-error_label)
            
            # get the test loss
            test_preds = self.forward(test_inputs)
            test_error_position = np.where(test_labels * test_preds < 0)
            test_error_input = test_inputs[test_error_position]
            test_error_label = test_labels[test_error_position]
            test_error_pred = test_preds[test_error_position]
            # calculate the loss
            test_loss = np.sum( -test_error_label * test_error_pred) / len(test_inputs)
            self.test_losses.append(test_loss)
        
        return self.losses
    
    def plot_loss(self):
        plt.clf()
        plt.plot(self.losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss', color='orange')
        plt.title('Loss curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("loss.png")
        pass
    
