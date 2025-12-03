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
        n_features = len(X[0])
        self.weight = np.zeros(n_features + 1)
        
        iterations = epochs if epochs is not None else self.max_iter

        if self.solver == "sgd":
            self._train_sgd(X, y, iterations)
        elif self.solver == "batch":
            self._train_batch(X, y, iterations)
        elif self.solver == "lbfgs":
            self._train_lbfgs(X, y, iterations)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
    
    def get_regularization_strength(self):
        n_samples = len(self.data)
        if n_samples == 0:
            return 0
        return 1.0 / (self.C * n_samples)


    # 1. Batch Logistic Regression
    def _train_batch(self, X, y, iterations):
        n_samples = len(X)
        lambda_reg = self.get_regularization_strength(n_samples)
        prev_weight = self.weight.copy()
        
        for epoch in range(iterations):
            grad = np.zeros(len(self.weight))

            for idx in range(n_samples):
                x, y_val = X[idx], y[idx]
                x_with_bias = np.concatenate([[self.bias], x])
                sigma = self.calculate_sigma(x)
                y_pred = self.calculate_probability(sigma)
                sample_weight = self.sample_weights_[idx]

                for i in range(len(self.weight)):
                    grad[i] += sample_weight * (y_val - y_pred) * x_with_bias[i]

            for i in range(len(self.weight)):
                if i > 0:
                    grad[i] -= lambda_reg * self.weight[i]
                self.weight[i] += self.learning_rate * grad[i]
            
            weight_diff = np.linalg.norm(self.weight - prev_weight)
            if weight_diff < self.tol:
                self.n_iter_ = epoch + 1
                break
            prev_weight = self.weight.copy()
        else:
            self.n_iter_ = iterations

    # 2. Stochastic Gradient Ascent
    # for all set of data randomized the order and train by that order until it gets the value 
    def _train_sgd(self, X, y, iterations):
        n_samples = len(X)
        lambda_reg = self.get_regularization_strength(n_samples)
        prev_weight = self.weight.copy()
        
        for epoch in range(iterations):
            indices = list(range(n_samples))
            random.shuffle(indices)

            for idx in indices:
                x, y_val = X[idx], y[idx]
                x_with_bias = np.concatenate([[self.bias], x])
                sigma = self.calculate_sigma(x)
                y_pred = self.calculate_probability(sigma)
                sample_weight = self.sample_weights_[idx]

                for j in range(len(self.weight)):
                    grad = sample_weight * (y_val - y_pred) * x_with_bias[j]
                    if j > 0:
                        grad -= lambda_reg * self.weight[j]
                    self.weight[j] += self.learning_rate * grad
            
            weight_diff = np.linalg.norm(self.weight - prev_weight)
            if weight_diff < self.tol:
                self.n_iter_ = epoch + 1
                break
            prev_weight = self.weight.copy()
        else:
            self.n_iter_ = iterations

    def predict(self, x):
        sigma = self.calculate_sigma(x)
        prob = self.calculate_probability(sigma)
        return 1 if prob > self.threshold else 0
    