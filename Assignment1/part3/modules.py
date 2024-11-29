import numpy as np

np.random.seed(42)

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.x = None
        self.number_of_samples = None
        self.params = {
            'weight': np.random.randn(in_features, out_features), 
            'bias': np.zeros((1, out_features))
            }
        self.grads = {'weight': None, 'bias': None}

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.x = x
        self.number_of_samples = x.shape[0]
        return np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        self.grads["weight"] = np.dot(self.x.T, dout)/ self.number_of_samples
        self.grads["bias"] = np.sum(dout, axis=0, keepdims=True)/ self.number_of_samples
        return np.dot(dout, self.params["weight"].T)

class ReLU(object):
    def __init__(self) -> None:
        self.x = None
    
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        return dout * (self.x > 0)

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, dout):
        """
        5
        Introduction to the Perceptron
        1 What is a Perceptron?
        A perceptron is a type of artificial neuron or the simplest form of a neural network, serving as the
        foundational building block for more complex neural networks. Introduced by Frank Rosenblatt in
        1957, it mimics the way a neuron in the brain works.
        2 How does a Perceptron Work?
        A perceptron takes multiple input values, each multiplied by a weight, sums them up, and produces
        a single binary output based on whether the sum is above a certain threshold.
        2.1 Mathematical Model
        The perceptron decision is based on these formulas:
        𝑓(𝑥)=𝑠𝑖𝑔𝑛(𝑤⋅𝑥+𝑏)
        𝑠𝑖𝑔𝑛(𝑥)={+1if𝑥≥0
        −1if𝑥<0
        Components of a Perceptron :
        1. Inputs (x): The features or data points provided to the perceptron.
        2. Weights (w): Coefficients determining the importance of each input.
        3. Bias (b): An extra input (always 1) with its own weight, allowing the activation function to shift,
        fitting the data better.
        4. Activation Function(sign function): Decides whether the neuron should activate, typically a
        step function for basic perceptrons.
        2.1.1 Geometric interpretation
        The perceptron divides the input space into two parts with a decision boundary, where for the inputs
        𝑥 that satisfy 𝑤·𝑥+𝑏>0, the predicted label is +1, and for those that satisfy 𝑤·𝑥+𝑏<0, the
        predicted label is −1. The decision boundary of the perceptron is a hyperplane defined by
        𝑤·𝑥+𝑏=0, which separates the space into two parts. On one side of the hyperplane, all the
        points are classified into one class, and on the other side, they are classified into another class. The
        direction of the vector w is perpendicular to the decision boundary, and the distance from the origin
        to the boundary is determined by the bias 𝑏.
        3 How to train a Perceptron
        3.1 Loss function
        Since we want to classify all points correctly, a natural idea is to directly use the total number of
        misclassified points as a loss function :
        The sign function cannot be differentiated, so the perceptron algorithm chooses the total distance
        from the misclassification point to the hyperplane S as the loss function:
        Regardless of coefficient  1
        ‖𝑤‖, we can get the final loss function of perceptron :
        The derivation of Loss function
        3.2 Gradient descent
        3.2.1 Introdcution
        How to update the parameters by minimizing the Loss function?
        if we want to minimize the function 𝑔(𝑥)=𝑥2, we can just use its derivation 𝑔′(𝑥)=0
        how about a more complex function?
        𝑓(𝑥,𝑦)=𝑥3−2𝑥2+𝑒𝑥𝑦−𝑦3+10𝑦2+100sin(𝑥𝑦)
        We take the partial derivatives of x and y respectively and set them to 0 to obtain the following
        system of equations:
        {3𝑥2−4𝑥+𝑦𝑒𝑥𝑦+100𝑦cos(𝑥𝑦)=0
        𝑥𝑒𝑥𝑦−3𝑦2+20𝑦+𝑥cos(𝑥𝑦)=0
        Hard to solve!
        When implemented in engineering, the iterative method is usually adopted, which starts from an
        initial point, repeatedly uses some rules to move from the next point, and constructs such a series
        until it converges to the point where the gradient is 0, that is, the gradient descent algorithm.
        Imagine a scenario where a person is stuck on a mountain (where the red circle is in the picture) and
        needs to get down from the mountain (to find the lowest point of the mountain, which is the valley),
        but the fog on the mountain is heavy at this time, resulting in poor visibility. Therefore, the path
        down the mountain cannot be determined, and he must use the information around him to find the
        path down the mountain. At this point, he can use the gradient descent algorithm to help him
        down the mountain. To be specific, take his current position as a benchmark, find the steepest
        place in this position (gradient), and then walk in the direction of the height of the mountain, and
        then every distance, and repeat the same method, and finally successfully reach the valley.
        The equation below describes what the gradient descent algorithm does: b is the next position of our
        climber, while a represents his current position. The minus sign refers to the minimization part of
        the gradient descent algorithm. The gamma in the middle is a waiting factor and the gradient term
        ▽𝑓(𝑎) is simply the direction of the steepest descent.
        𝑏=𝑎−𝛾▽𝑓(𝑎)
        3.2.2 Select appropriate Learning Rate
        For the gradient descent algorithm to reach the local minimum we must set the learning rate to an
        appropriate value, which is neither too low nor too high. This is important because if the steps it
        takes are too big, it may not reach the local minimum because it bounces back and forth between the
        convex function of gradient descent (see left image below). If we set the learning rate to a very small
        value, gradient descent will eventually reach the local minimum but that may take a while (see the
        right image)
        3.2.3 Types of Gradient Descent
        • Batch gradient descent, also called vanilla gradient descent, calculates the error for each
        example within the training dataset, but only after all training examples have been evaluated
        does the model get updated. This whole process is like a cycle and it’s called a training epoch.
        Some advantages of batch gradient descent are its computational efficiency: it produces a stable
        error gradient and a stable convergence. Some disadvantages are that the stable error gradient can
        sometimes result in a state of convergence that isn’t the best the model can achieve. It also
        requires the entire training dataset to be in memory and available to the algorithm.
        for i in range(nb_epochs):
          params_grad = evaluate_gradient(loss_function,data,params)
          params = params - learning_rate * params_grad
        • Stochastic gradient descent (SGD), by contrast, does this for each training example within
        the dataset, meaning it updates the parameters for each training example one by one. Depending
        on the problem, this can make SGD faster than batch gradient descent. One advantage is the
        frequent updates allow us to have a pretty detailed rate of improvement.
        The frequent updates, however, are more computationally expensive than the batch gradient
        descent approach. Additionally, the frequency of those updates can result in noisy gradients,
        which may cause the error rate to jump around instead of slowly decreasing.
        for i in range(nb_epochs):
           np.random.shuffle(data)
           for example in data:
              params_grad = evaluate_gradient(loss_functon,example,params)
              params = params - learning_rate * params_grad
        
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout

class CrossEntropy(object):
    
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        epsilon = 1e-10
        return -np.sum(y * np.log(x + epsilon))

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        return x - y
