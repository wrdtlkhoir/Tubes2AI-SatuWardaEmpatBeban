import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer
import math
import random
from scipy.optimize import minimize
from scipy.special import logsumexp

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
        x_with_bias = np.concatenate([[self.bias], x])
        return np.dot(self.weight, x_with_bias)
    
    def calculate_probability(self, sigma):
        sigma = np.clip(sigma, -500, 500)
        if sigma >= 0:
            z = np.exp(-sigma)
            return 1 / (1 + z)
        else:
            z = np.exp(sigma)
            return z / (1 + z)
    
    def softmax(self, z):
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y, epochs=None):
        X = np.asarray(X)
        y_original = np.asarray(y)
        y = y_original.copy()
        
        if y.dtype.kind not in ['i', 'u', 'f']:
            unique_labels = np.unique(y)
            self.label_mapping_ = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            self.inverse_label_mapping_ = {idx: label for label, idx in self.label_mapping_.items()}
            y = np.array([self.label_mapping_[label] for label in y])
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.sample_weights_ = self._compute_class_weights(y)
        
        # Initialize weights
        n_samples, n_features = X.shape
        if self.solver != "lbfgs":
            self.weight = np.zeros(n_features + 1)
        
        # Set random seed
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
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
    def _train_lbfgs(self, X, y, iterations):
        n_samples, n_features = X.shape
        n_classes = self.n_classes_
        
        encoder = OneHotEncoder(sparse_output=False, categories=[self.classes_])
        Y_onehot = encoder.fit_transform(y.reshape(-1, 1))
        
        sample_weights = self.sample_weights_
        
        # Regularization
        alpha = 1.0 / self.C
        
        def loss_grad(params):
            """Compute loss and gradient for L-BFGS-B."""
            # Reshape
            W = params[:n_features * n_classes].reshape(n_features, n_classes)
            b = params[n_features * n_classes:]
            
            Z = X @ W + b
            
            # log probabilities
            lse = logsumexp(Z, axis=1, keepdims=True)
            log_probs = Z - lse
            probs = np.exp(log_probs)
            
            # Compute loss
            if sample_weights is not None:
                loss_term = -np.sum(sample_weights[:, np.newaxis] * Y_onehot * log_probs)
            else:
                loss_term = -np.sum(Y_onehot * log_probs)
            
            reg_term = 0.5 * alpha * np.sum(W**2)
            total_loss = loss_term + reg_term
            
            # Compute gradients
            diff = probs - Y_onehot
            if sample_weights is not None:
                diff = diff * sample_weights[:, np.newaxis]
            
            grad_W = X.T @ diff + alpha * W
            grad_b = np.sum(diff, axis=0)
            grad = np.concatenate([grad_W.ravel(), grad_b])
            
            return total_loss, grad
        
        # Initialize parameters
        initial_params = np.zeros(n_features * n_classes + n_classes)
        
        # Optimize
        res = minimize(
            fun=loss_grad,
            x0=initial_params,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': iterations, 'gtol': self.tol}
        )
        
        # Extract optimized parameters
        self.coef_ = res.x[:n_features * n_classes].reshape(n_features, n_classes).T
        self.intercept_ = res.x[n_features * n_classes:]
        self.n_iter_ = res.nit

    @property
    def feature_importances_(self):
        if self.weight is None:
            raise ValueError("Model must be fitted before accessing feature importances")
        
        # Use L-BFGS-B coefficients
        if hasattr(self, 'coef_') and hasattr(self, 'intercept_'):
            if self.coef_.ndim > 1:
                importances = np.linalg.norm(self.coef_, axis=0)
            else:
                importances = np.abs(self.coef_)
        else:
            # Use standard weights for SGD/Batch
            importances = np.abs(self.weight[1:])
        
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        return importances
    
    def get_params(self, deep=True):
        return {
            'solver': self.solver,
            'learning_rate': self.learning_rate,
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'tol': self.tol
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
        
        if 'random_state' in params and params['random_state'] is not None:
            random.seed(params['random_state'])
            np.random.seed(params['random_state'])
        
        return self
    

    def predict(self, x):
        sigma = self.calculate_sigma(x)
        prob = self.calculate_probability(sigma)
        return 1 if prob > self.threshold else 0
    