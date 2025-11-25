import numpy as np
import math
import random

class LogRegression():
    def __init__(self):
        weight = None
        data = [] # List of (feature vectors, labels)
        threshold = 0.5
        bias = 1
        learning_rate = 0.5

    # Setter Getter
    def set_bias(self, value):
        self.bias = value

    def set_threshold(self, value):
        self.threshold = value
    
    def set_learning_rate(self, value):
        self.learning_rate = value
   
    # Core Function
    # X = [[1,2,3], [4,5,6], ...]
    # y = [1, 0, 1, ...]
    def add_data(self, X, y):
        for i in range (len(X)):
            self.data.append((X[i], y[i]))
        
        # Initialize weights
        if self.weight is None: 
            n_features = len(X[0])
            self.weight = np.zeros(n_features + 1)

    def calculate_sigma(self, x):
        x_with_bias = [self.bias] + x
        return np.dot(self.weight, x_with_bias)
    
    def calculate_probability(self, sigma):
        return 1 / (1 + pow(math.e, -sigma))
    
    # for all set of data randomized the order and iterate by that order until it gets the value 
    def iterate(self, epochs=10):
        for epoch in range(epoch): 
            random.shuffle(self.data)

            for x, y in self.data:
                x_with_bias = [self.bias] + x

                sigma = self.calculate_sigma(x)
                y_pred = self.calculate_probability(sigma)

                for i in range(len(self.weight)):
                    self.weight[i] += self.learning_rate * (y - y_pred) * x_with_bias[i]

    def predict(self, x):
        sigma = self.calculate_sigma(x)
        prob = self.calculate_probability(sigma)
        return 1 if prob > self.threshold else 0
    