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
    
    # 1. Batch Logistic Regression
    def iterate_batch(self, epochs=10):
        for epoch in range(epochs):
            grad = np.zeros(len(self.weight))

            for x, y in self.data: 
                x_with_bias = [self.bias] + x

                sigma = self.calculate_sigma(x)
                y_pred = self.calculate_probability(sigma)

                for i in range (len(self.weight)):
                    grad[i] += (y - y_pred) * x_with_bias[i]

            for i in range(len(self.weight)):
                self.weight[i] += self.learning_rate * grad[i]

    # 2. Stochastic Gradient Ascent
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

    #3. Mini batch gradient ascent
    def iterate_mini_batch(self, batch_size=4, epochs=10):
        for epoch in range(epoch): 
            random.shuffle(self.data)

            for i in range(0, len(self.data), batch_size):
                batch = self.data[i:i+batch_size]

                # Initialize batch gradient
                grad = np.zeros(len(self.weight))

                for x, y in batch:
                    x_with_bias = [self.bias] + x
                    sigma = self.calculate_sigma(x)
                    y_pred = self.calculate_probability(sigma)

                    # Accumulate gradient
                    for j in range(len(self.weight)):
                        grad[j] += (y - y_pred) * x_with_bias[j]

                # Update weight per batch
                for j in range(self.weight):
                    self.weight[j] += self.learning_rate * grad[j]

    def predict(self, x):
        sigma = self.calculate_sigma(x)
        prob = self.calculate_probability(sigma)
        return 1 if prob > self.threshold else 0
    