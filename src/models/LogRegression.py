import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

class LogRegression():
    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.1, random_state=None, solver='lbfgs', class_weight=None, tol=1e-4 ):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.solver = solver
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

        # Bonus
        self.loss_history_ = []
        self.param_history_ = []

    def _compute_loss(self, X, y, Y_onehot=None):
        n_samples = len(X)
        lambda_reg = self.get_regularization_strength(n_samples)
        
        if Y_onehot is None:
            Y_onehot = np.zeros((n_samples, self.n_classes_))
            Y_onehot[np.arange(n_samples), y] = 1
        
        Z = X @ self.coef_ + self.intercept_
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(Z[:, 1] - Z[:, 0]).reshape(-1, 1)
            probs = np.hstack([1 - probs, probs])
        else:
            probs = self.softmax(Z)
        
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(Y_onehot * np.log(probs) * self.sample_weights_[:, np.newaxis], axis=1))
        loss += 0.5 * lambda_reg * np.sum(self.coef_**2)
        
        return loss

    # Setter Getter
    def set_learning_rate(self, value):
        self.learning_rate = value
   
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
        
        self.coef_ = np.zeros((n_features, self.n_classes_))
        self.intercept_ = np.zeros(self.n_classes_)
        
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
    
    def get_regularization_strength(self, n_samples):
        if n_samples == 0:
            return 0
        return 1.0 / (self.C * n_samples)

    # 1. Batch Logistic Regression
    def _train_batch(self, X, y, iterations):
        n_samples = len(X)
        lambda_reg = self.get_regularization_strength(n_samples)
        
        Y_onehot = np.zeros((n_samples, self.n_classes_))
        Y_onehot[np.arange(n_samples), y] = 1
        
        prev_coef = self.coef_.copy()
        prev_intercept = self.intercept_.copy()
        
        # Initialize tracking
        self.loss_history_ = []
        self.param_history_ = []
        
        for epoch in range(iterations):
            if self.n_classes_ == 2:
                self.param_history_.append([self.intercept_[1], self.coef_[0, 1]])
            else:
                # multinomial
                self.param_history_.append(np.concatenate([self.coef_.ravel(), self.intercept_]))
            loss = self._compute_loss(X, y, Y_onehot)
            self.loss_history_.append(loss)
            
            Z = X @ self.coef_ + self.intercept_
            
            if self.n_classes_ == 2:
                probs = self._sigmoid(Z[:, 1] - Z[:, 0]).reshape(-1, 1)
                probs = np.hstack([1 - probs, probs])
            else:
                probs = self.softmax(Z)
            
            error = probs - Y_onehot
            weighted_error = error * self.sample_weights_[:, np.newaxis]
            
            grad_coef = (X.T @ weighted_error) / n_samples + lambda_reg * self.coef_
            grad_intercept = np.sum(weighted_error, axis=0) / n_samples
            
            # Update parameters
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
            
            # Check convergence
            coef_diff = np.linalg.norm(self.coef_ - prev_coef)
            intercept_diff = np.linalg.norm(self.intercept_ - prev_intercept)
            if coef_diff + intercept_diff < self.tol:
                self.n_iter_ = epoch + 1
                break
            
            prev_coef = self.coef_.copy()
            prev_intercept = self.intercept_.copy()
        else:
            self.n_iter_ = iterations

    # 2. Stochastic Gradient Descent
    def _train_sgd(self, X, y, iterations):
        n_samples = len(X)
        lambda_reg = self.get_regularization_strength(n_samples)
        
        prev_coef = self.coef_.copy()
        prev_intercept = self.intercept_.copy()
        
        self.loss_history_ = []
        self.param_history_ = []
        
        for epoch in range(iterations):
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for idx in indices:
                if self.n_classes_ == 2:
                    self.param_history_.append([self.intercept_[1], self.coef_[0, 1]])
                else:
                    self.param_history_.append(np.concatenate([self.coef_.ravel(), self.intercept_]))
                
                if len(self.loss_history_) % 10 == 0: 
                    loss = self._compute_loss(X, y)
                    self.loss_history_.append(loss)
                
                x_i = X[idx:idx+1]  
                y_i = y[idx]
                
                z = x_i @ self.coef_ + self.intercept_
                
                if self.n_classes_ == 2:
                    prob = self._sigmoid(z[0, 1] - z[0, 0])
                    probs = np.array([[1 - prob, prob]])
                else:
                    probs = self.softmax(z)
                
                y_onehot = np.zeros((1, self.n_classes_))
                y_onehot[0, y_i] = 1
                error = (probs - y_onehot) * self.sample_weights_[idx]
                
                grad_coef = x_i.T @ error + lambda_reg * self.coef_
                grad_intercept = error[0]
                
                # Update parameters
                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept
            
            # Check convergence
            coef_diff = np.linalg.norm(self.coef_ - prev_coef)
            intercept_diff = np.linalg.norm(self.intercept_ - prev_intercept)
            if coef_diff + intercept_diff < self.tol:
                self.n_iter_ = epoch + 1
                break
            
            prev_coef = self.coef_.copy()
            prev_intercept = self.intercept_.copy()
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
            W = params[:n_features * n_classes].reshape(n_features, n_classes)
            b = params[n_features * n_classes:]
            
            Z = X @ W + b
            
            lse = logsumexp(Z, axis=1, keepdims=True)
            log_probs = Z - lse
            probs = np.exp(log_probs)
            
            if sample_weights is not None:
                loss_term = -np.sum(sample_weights[:, np.newaxis] * Y_onehot * log_probs)
            else:
                loss_term = -np.sum(Y_onehot * log_probs)
            
            reg_term = 0.5 * alpha * np.sum(W**2)
            total_loss = loss_term + reg_term
            
            diff = probs - Y_onehot
            if sample_weights is not None:
                diff = diff * sample_weights[:, np.newaxis]
            
            grad_W = X.T @ diff + alpha * W
            grad_b = np.sum(diff, axis=0)
            grad = np.concatenate([grad_W.ravel(), grad_b])
            
            return total_loss, grad
        
        initial_params = np.zeros(n_features * n_classes + n_classes)
        
        self.loss_history_ = []
        self.param_history_ = []
        
        def callback(params):
            W = params[:n_features * n_classes].reshape(n_features, n_classes)
            b = params[n_features * n_classes:]
            
            if self.n_classes_ == 2:
                self.param_history_.append([b[1], W[0, 1]])
            else:
                self.param_history_.append(params.copy())
            
            loss, _ = loss_grad(params)
            self.loss_history_.append(loss)
        
        res = minimize(
            fun=loss_grad,
            x0=initial_params,
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options={'maxiter': iterations, 'gtol': self.tol}
        )
        
        self.coef_ = res.x[:n_features * n_classes].reshape(n_features, n_classes).T
        self.intercept_ = res.x[n_features * n_classes:]
        self.n_iter_ = res.nit
    
    def _compute_class_weights(self, y_encoded):
        if self.class_weight is None:
            return np.ones(len(y_encoded))
        elif self.class_weight == 'balanced':
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            n_samples = len(y_encoded)
            n_classes = len(unique_classes)
            
            class_weight_dict = {}
            for cls, count in zip(unique_classes, class_counts):
                class_weight_dict[cls] = n_samples / (n_classes * count)
            
            return np.array([class_weight_dict[cls] for cls in y_encoded])
        elif isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(cls, 1.0) for cls in y_encoded])
        else:
            raise ValueError("class_weight must be None, 'balanced', or a dictionary")

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
    

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        Z = X @ (self.coef_.T if self.solver=='lbfgs' else self.coef_) + self.intercept_
        
        if self.n_classes_ == 2:
            probs_1 = self._sigmoid(Z[:, 1] - Z[:, 0]).reshape(-1, 1)
            probs = np.hstack([1 - probs_1, probs_1])
        else:
            probs = self.softmax(Z)
        
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        
        if hasattr(self, 'inverse_label_mapping_'):
            predictions = np.array([self.inverse_label_mapping_[p] for p in predictions])
        
        return predictions

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    
    def animate_logloss(self, X, y, save_path='logloss_animation.gif', fps=10, param_indices=(0, 1)):
        if len(self.param_history_) == 0:
            raise ValueError("No training history found.")
        
        params = np.array(self.param_history_)
        theta1_path = params[:, param_indices[0]]
        theta2_path = params[:, param_indices[1]]
        
        param1_name = f'θ_{param_indices[0]}'
        param2_name = f'θ_{param_indices[1]}'
        
        # 30x30 grid
        theta1_range = theta1_path.max() - theta1_path.min()
        theta2_range = theta2_path.max() - theta2_path.min()
        theta1_min = theta1_path.min() - 0.2 * theta1_range
        theta1_max = theta1_path.max() + 0.2 * theta1_range
        theta2_min = theta2_path.min() - 0.2 * theta2_range
        theta2_max = theta2_path.max() + 0.2 * theta2_range
        
        theta_grid_1 = np.linspace(theta1_min, theta1_max, 30)
        theta_grid_2 = np.linspace(theta2_min, theta2_max, 30)
        Theta1, Theta2 = np.meshgrid(theta_grid_1, theta_grid_2)
        
        # Compute loss surface
        Loss = np.zeros_like(Theta1)
        n_samples = len(X)
        
        current_W = self.coef_.T.copy() if self.solver == 'lbfgs' else self.coef_.copy()
        current_b = self.intercept_.copy()
        n_features = current_W.shape[0]
        n_coefs = n_features * self.n_classes_
        
        for i in range(30):
            for j in range(30):
                temp_W = current_W.copy()
                temp_b = current_b.copy()
                
                # Update parameters
                if param_indices[0] < n_coefs:
                    f1, c1 = param_indices[0] // self.n_classes_, param_indices[0] % self.n_classes_
                    temp_W[f1, c1] = Theta1[i, j]
                else:
                    temp_b[param_indices[0] - n_coefs] = Theta1[i, j]
                
                if param_indices[1] < n_coefs:
                    f2, c2 = param_indices[1] // self.n_classes_, param_indices[1] % self.n_classes_
                    temp_W[f2, c2] = Theta2[i, j]
                else:
                    temp_b[param_indices[1] - n_coefs] = Theta2[i, j]
                
                # Compute loss
                Z = X @ temp_W + temp_b
                probs = self.softmax(Z)
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                
                Y_onehot = np.zeros((n_samples, self.n_classes_))
                Y_onehot[np.arange(n_samples), y] = 1
                
                Loss[i, j] = -np.mean(np.sum(Y_onehot * np.log(probs), axis=1))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contour(Theta1, Theta2, Loss, levels=15, cmap='viridis')
        ax.contourf(Theta1, Theta2, Loss, levels=15, cmap='viridis', alpha=0.3)
        
        line, = ax.plot([], [], 'r-', linewidth=2)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title('Loss Landscape and Training Path')
        
        def animate(frame):
            line.set_data(theta1_path[:frame+1], theta2_path[:frame+1])
            point.set_data([theta1_path[frame]], [theta2_path[frame]])
            return line, point
        
        anim = animation.FuncAnimation(fig, animate, frames=len(theta1_path), interval=1000//fps, blit=True)
        
        print(f"Saving to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        
        return anim
    
class RFE:
    def __init__(self, estimator, n_features_to_select=None, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.support_ = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select
            
        support = np.ones(n_features, dtype=bool)
        current_features = n_features
        
        while current_features > n_features_to_select:
            X_subset = X[:, support]
            
            params = self.estimator.get_params()
            model = LogRegression(**params)
            model.fit(X_subset, y)
            
            if model.coef_.ndim > 1:
                importances = np.linalg.norm(model.coef_, axis=0)
            else:
                importances = np.abs(model.coef_)
                
            if isinstance(self.step, float):
                step_size = max(1, int(self.step * n_features))
            else:
                step_size = self.step
                
            n_to_remove = min(step_size, current_features - n_features_to_select)
            indices = np.argsort(importances)[:n_to_remove]
            features_idx_map = np.where(support)[0]
            features_to_remove = features_idx_map[indices]
            
            support[features_to_remove] = False
            current_features -= n_to_remove
            print(f"RFE: {current_features} features left...")
            
        self.support_ = support
        return self

    def transform(self, X):
        return X[:, self.support_]


