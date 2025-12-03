import numpy as np
import math
import random

class LogRegression():
    def __init__(self, C=1.0, max_iter=1000, random_state=None, solver='lbfgs', class_weight=None ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.class_weight = class_weight
        self.weight = None
        self.threshold = 0.5
        self.bias = 1
        self.learning_rate = 0.5
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    # Setter Getter
    def set_bias(self, value):
        self.bias = value

    def set_threshold(self, value):
        self.threshold = value
    
    def set_learning_rate(self, value):
        self.learning_rate = value
   
    def calculate_sigma(self, x):
        x_with_bias = [self.bias] + x
        return np.dot(self.weight, x_with_bias)
    
    def calculate_probability(self, sigma):
        return 1 / (1 + pow(math.e, -sigma))
    
    def softmax(self, z):
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y, epochs=None):
        self.data = []
        self.add_data(X, y)
        
        n_features = len(X[0])
        self.weight = np.zeros(n_features + 1)
        
        iterations = epochs if epochs is not None else self.max_iter

        if self.solver == "sgd":
            self._train_sgd(iterations)
        elif self.solver == "batch":
            self._train_batch(iterations)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
    


    # 1. Batch Logistic Regression
    def train_batch(self, epochs=10):
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
    # for all set of data randomized the order and train by that order until it gets the value 
    def train_sdg(self, epochs=10):
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
    