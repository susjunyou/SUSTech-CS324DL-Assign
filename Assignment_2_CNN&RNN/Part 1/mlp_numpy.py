from modules import * 

class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes the multi-layer perceptron object.
        
        This function should initialize the layers of the MLP including any linear layers and activation functions 
        you plan to use. You will need to create a list of linear layers based on n_inputs, n_hidden, and n_classes.
        Also, initialize ReLU activation layers for each hidden layer and a softmax layer for the output.
        
        Args:
            n_inputs (int): Number of inputs (i.e., dimension of an input vector).
            n_hidden (list of int): List of integers, where each integer is the number of units in each hidden layer.
            n_classes (int): Number of classes of the classification problem (i.e., output dimension of the network).
        """
        # Hint: You can use a loop to create the necessary number of layers and add them to a list.
        # Remember to initialize the weights and biases in each layer.
        self.layers = []
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        self.n_layers = len(n_hidden)
        self.layers.append(Linear(n_inputs, n_hidden[0]))
        self.layers.append(ReLU())
        for i in range(self.n_layers - 1):
            self.layers.append(Linear(n_hidden[i], n_hidden[i + 1]))
            self.layers.append(ReLU())
        self.layers.append(Linear(n_hidden[-1], n_classes))
        self.layers.append(SoftMax())
        
        
    def forward(self, x):
        """
        Predicts the network output from the input by passing it through several layers.
        
        Here, you should implement the forward pass through all layers of the MLP. This involves
        iterating over your list of layers and passing the input through each one sequentially.
        Don't forget to apply the activation function after each linear layer except for the output layer.
        
        Args:
            x (numpy.ndarray): Input to the network.
            
        Returns:
            numpy.ndarray: Output of the network.
        """
        # Start with the input as the initial output
        out = x  
        
        # TODO: Implement the forward pass through each layer.
        # Hint: For each layer in your network, you will need to update 'out' to be the layer's output.
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Performs the backward propagation pass given the loss gradients.
        
        Here, you should implement the backward pass through all layers of the MLP. This involves
        iterating over your list of layers in reverse and passing the gradient through each one sequentially.
        You will update the gradients for each layer.
        
        Args:
            dout (numpy.ndarray): Gradients of the loss with respect to the output of the network.
        """
        # TODO: Implement the backward pass through each layer.
        # Hint: You will need to update 'dout' to be the gradient of the loss with respect to the input of each layer.
        
        # No need to return anything since the gradients are stored in the layers.
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def update(self, lr):
        """
        Updates the weights of the network using the previously calculated gradients.
        
        This function should update the weights of each linear layer in the network based on the stored gradients.
        
        Args:
            lr (float): Learning rate for the update.
        """
        for layer in self.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                layer.params["weight"] -= lr * layer.grads["weight"]
                layer.params["bias"] -= lr * layer.grads["bias"]        
                
                
