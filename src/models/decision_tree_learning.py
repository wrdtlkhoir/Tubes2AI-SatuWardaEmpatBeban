import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Union, Any


class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        self.children = {}
        self.is_leaf = False
        self.class_distribution = None
        
        
class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='entropy', random_state=None, class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.class_weight = class_weight
        self.tree = None
        self.feature_names = None
        self.classes = None
        self.classes_ = None  # Untuk sklearn compatibility
        self.rng = None
        self.sample_weights = None
        self.feature_importances_ = None  # Untuk RFE
        
    def fit(self, X, y):
        if self.random_state is not None:
            self.rng = np.random.RandomState(self.random_state)
        else:
            self.rng = np.random.RandomState()
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"X{i}" for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            y = y.values
            
        self.classes = np.unique(y)
        self.classes_ = self.classes
        self.sample_weights = self._compute_sample_weights(y)
        self.tree = self.build_tree(X, y, depth=0, weights=self.sample_weights)
        self.feature_importances_ = self._compute_feature_importance(X, y)
        return self

    def _compute_sample_weights(self, y):
        n_samples = len(y)
        weights = np.ones(n_samples, dtype=np.float64)
        
        if self.class_weight == 'balanced':
            unique_classes, counts = np.unique(y, return_counts=True)
            n_classes = len(unique_classes)
            
            # balanced weight = n_samples / (n_classes * n_samples_for_class)
            class_weights = {}
            for cls, count in zip(unique_classes, counts):
                class_weights[cls] = n_samples / (n_classes * count)
    
            for i, cls in enumerate(y):
                weights[i] = class_weights[cls]
                
        elif isinstance(self.class_weight, dict):
            for i, cls in enumerate(y):
                weights[i] = self.class_weight.get(cls, 1.0)
        
        return weights
    
    def is_continuous(self, x):
        try:
            if isinstance(x, pd.Series):
                x_clean = x.dropna().values
            else:
                x_clean = x[~np.isnan(x)] if np.issubdtype(x.dtype, np.number) else x
            
            if len(x_clean) == 0:
                return False
            
            # Check if numeric
            if not np.issubdtype(type(x_clean[0]), np.number):
                return False
            
            unique_count = len(np.unique(x_clean))
            return unique_count > 5
        except:
            return False
    
    def entropy(self, y, weights=None):
        if weights is None:
            weights = np.ones(len(y))
            
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0
        
        unique_classes = np.unique(y)
        entropy_val = 0
        
        for cls in unique_classes:
            mask = y == cls
            weight_sum = np.sum(weights[mask])
            if weight_sum > 0:
                p = weight_sum / total_weight
                entropy_val -= p * np.log2(p)
        
        return entropy_val
    
    def information_gain(self, X, y, feature_idx, threshold=None, weights=None):
        if weights is None:
            weights = np.ones(len(y))
            
        parent_entropy = self.entropy(y, weights)
        
        mask = ~np.isnan(X[:, feature_idx])
        if mask.sum() == 0:
            return 0
        
        X_valid = X[mask]
        y_valid = y[mask]
        weights_valid = weights[mask]
        
        if threshold is not None:
            left_mask = X_valid[:, feature_idx] <= threshold
            right_mask = ~left_mask
        
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                return 0
            splits = [
                (y_valid[left_mask], weights_valid[left_mask]),
                (y_valid[right_mask], weights_valid[right_mask])
            ]
        else:
            values = np.unique(X_valid[:, feature_idx])
            if len(values) < 2:
                return 0
            splits = [
                (y_valid[X_valid[:, feature_idx] == v], 
                 weights_valid[X_valid[:, feature_idx] == v])
                for v in values
            ]
        
        total_weight = np.sum(weights_valid)
        weighted_entropy = 0
        
        for split_y, split_weights in splits:
            if len(split_y) > 0:
                weight = np.sum(split_weights) / total_weight
                weighted_entropy += weight * self.entropy(split_y, split_weights)
        
        gain = parent_entropy - weighted_entropy
        return max(0,gain)
    
    def gain_ratio(self, X, y, feature_idx, threshold=None, weights=None):
        gain = self.information_gain(X, y, feature_idx, threshold, weights)

        if gain <= 1e-10:
            return 0
        
        mask = ~np.isnan(X[:, feature_idx])
        if mask.sum() == 0:
            return 0
        
        X_valid = X[mask]
        weights_valid = weights[mask] if weights is not None else np.ones(mask.sum())
        
        if threshold is not None:
            left_mask = X_valid[:, feature_idx] <= threshold
            right_mask = ~left_mask
            splits = [np.sum(weights_valid[left_mask]), np.sum(weights_valid[right_mask])]
        else:
            values = np.unique(X_valid[:, feature_idx])
            splits = [np.sum(weights_valid[X_valid[:, feature_idx] == v]) for v in values]
        
        total_weight = np.sum(weights_valid)
        split_info = 0
        
        for weight_sum in splits:
            if weight_sum > 0:
                proportion = weight_sum / total_weight
                split_info -= proportion * np.log2(proportion)
        
        # Avoid division by very small split_info (use small epsilon)
        if split_info < 1e-10:
            return gain * 0.5
        return gain / split_info
    
    def find_best_split(self, X, y, weights):
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_candidates = []
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            x_feature = X[:, feature_idx]
            
            if np.all(np.isnan(x_feature)):
                continue
            
            if self.is_continuous(x_feature):
                mask_valid = ~np.isnan(x_feature)
                x_clean = x_feature[mask_valid]
                y_clean = y[mask_valid]
                weights_clean = weights[mask_valid]
                
                sorted_indices = np.argsort(x_clean)
                x_sorted = x_clean[sorted_indices]
                y_sorted = y_clean[sorted_indices]
                
                candidate_thresholds = []
                for i in range(len(y_sorted) - 1):
                    if y_sorted[i] != y_sorted[i + 1]:
                        threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                        candidate_thresholds.append(threshold)
                        
                if len(candidate_thresholds) > 100:
                    step = len(candidate_thresholds) // 100
                    candidate_thresholds = candidate_thresholds[::step]
                
                for threshold in candidate_thresholds:
                    if self.criterion == 'gain_ratio':
                        gain = self.gain_ratio(X, y, feature_idx, threshold, weights)
                    else:
                        gain = self.information_gain(X, y, feature_idx, threshold, weights)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [(feature_idx, threshold)]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append((feature_idx, threshold))
            else:
                if self.criterion == 'gain_ratio':
                    gain = self.gain_ratio(X, y, feature_idx, None, weights)
                else:
                    gain = self.information_gain(X, y, feature_idx, None, weights)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [(feature_idx, None)]
                elif gain == best_gain and gain > 0:
                    best_candidates.append((feature_idx, None))
        if best_candidates:
            chosen_idx = self.rng.randint(0, len(best_candidates))
            best_feature, best_threshold = best_candidates[chosen_idx]
        
        return best_feature, best_threshold
    
    def majority_class(self, y, weights=None):
        if weights is None:
            return np.bincount(y).argmax()
        
        # Weighted majority class
        unique_classes = np.unique(y)
        class_weights = np.zeros(len(unique_classes))
        
        for i, cls in enumerate(unique_classes):
            class_weights[i] = np.sum(weights[y == cls])
        
        return unique_classes[np.argmax(class_weights)]
    
    def build_tree(self, X, y, depth, weights=None):
        if weights is None:
            weights = np.ones(len(y))
            if self.sample_weights is not None and len(self.sample_weights) >= len(y):
                weights = self.sample_weights[:len(y)]
                
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        node = Node()
        node.class_distribution = np.bincount(y, weights=weights, minlength=len(self.classes))
        
        if (n_classes == 1 or 
            n_samples < self.min_samples_split or
            (self.max_depth and depth >= self.max_depth)):
            node.is_leaf = True
            node.value = self.classes[self.majority_class(y)]
            return node
        
        best_feature, best_threshold = self.find_best_split(X, y, weights)
        
        if best_feature is None:
            node.is_leaf = True
            node.value = self.classes[self.majority_class(y, weights)]
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        node.value = self.classes[self.majority_class(y, weights)] 
        
        x_feature = X[:, best_feature]
        
        if best_threshold is not None:
            mask_valid = ~np.isnan(x_feature)
            left_mask = np.zeros(n_samples, dtype=bool)
            right_mask = np.zeros(n_samples, dtype=bool)
            
            left_mask[mask_valid] = x_feature[mask_valid] <= best_threshold
            right_mask[mask_valid] = x_feature[mask_valid] > best_threshold
            
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left >= n_right:
                left_mask[~mask_valid] = True
            else:
                right_mask[~mask_valid] = True
            
            if left_mask.sum() >= self.min_samples_leaf:
                node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1, weights[left_mask])
            else:
                node.left = Node()
                node.left.is_leaf = True
                node.left.value = self.classes[self.majority_class(y, weights)]
                node.left.class_distribution = node.class_distribution.copy()
                
            if right_mask.sum() >= self.min_samples_leaf:
                node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1, weights[right_mask])
            else:
                node.right = Node()
                node.right.is_leaf = True
                node.right.value = self.classes[self.majority_class(y, weights)]
                node.right.class_distribution = node.class_distribution.copy()
                
        else:
            values = np.unique(x_feature[~np.isnan(x_feature)])
            
            for value in values:
                mask = x_feature == value
                if mask.sum() >= self.min_samples_leaf:
                    node.children[value] = self.build_tree(X[mask], y[mask], depth + 1, weights[mask])
                else:
                    child = Node()
                    child.is_leaf = True
                    if mask.sum() > 0:
                        child.value = self.classes[self.majority_class(y[mask], weights[mask])]
                        child.class_distribution = np.bincount(y[mask], weights=weights[mask], minlength=len(self.classes))
                    else:
                        child.value = self.classes[self.majority_class(y, weights)]
                        child.class_distribution = node.class_distribution.copy()
                    node.children[value] = child
            
            if len(node.children) == 0:
                node.is_leaf = True
                node.value = self.classes[self.majority_class(y, weights)]
        
        return node
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self.predict_sample(x, self.tree) for x in X])
    
    def predict_sample(self, x, node):
        if node.is_leaf:
            return node.value
        
        feature_val = x[node.feature]
        
        if np.isnan(feature_val) if isinstance(feature_val, (int, float)) else pd.isna(feature_val):
            if node.value is not None:
                return node.value
            if node.class_distribution is not None:
                return self.classes[np.argmax(node.class_distribution)]
            return self.classes[0]
        
        if node.threshold is not None:
            # Binary split
            if feature_val <= node.threshold:
                if node.left:
                    return self.predict_sample(x, node.left)
                else:
                    return node.value if node.value is not None else self.classes[0]
            else:
                if node.right:
                    return self.predict_sample(x, node.right)
                else:
                    return node.value if node.value is not None else self.classes[0]
        else:
            # Categorical feature
            if feature_val in node.children:
                return self.predict_sample(x, node.children[feature_val])
            else:
                # If value not seen in training, use majority class at this node
                return node.value if node.value is not None else self.classes[0]
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        probs = np.zeros((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            node = self._get_leaf_node(x, self.tree)
            if node and node.class_distribution is not None:
                total = np.sum(node.class_distribution)
                if total > 0:
                    probs[i] = node.class_distribution / total
                else:
                    probs[i] = np.ones(len(self.classes_)) / len(self.classes_)
            else:
                probs[i] = np.ones(len(self.classes_)) / len(self.classes_)
        return probs

    def _get_leaf_node(self, x, node):
        if node is None or node.is_leaf:
            return node
        
        feature_val = x[node.feature]
        
        if pd.isna(feature_val):
            return node
        
        if node.threshold is not None:
            if feature_val <= node.threshold:
                return self._get_leaf_node(x, node.left) if node.left else node
            else:
                return self._get_leaf_node(x, node.right) if node.right else node
        else:
            if feature_val in node.children:
                return self._get_leaf_node(x, node.children[feature_val])
            else:
                return node

    def _compute_feature_importance(self, X, y):
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        def traverse(node, n_samples):
            if node is None or node.is_leaf:
                return
            
            if node.feature is not None:
                # Hitung information gain untuk split ini
                gain = self.information_gain(X, y, node.feature, node.threshold)
                importances[node.feature] += gain * n_samples
            
            # Rekursi ke children
            if node.threshold is not None:
                if node.left:
                    n_left = np.sum((X[:, node.feature] <= node.threshold) & (~pd.isna(X[:, node.feature])))
                    traverse(node.left, n_left)
                if node.right:
                    n_right = np.sum((X[:, node.feature] > node.threshold) & (~pd.isna(X[:, node.feature])))
                    traverse(node.right, n_right)
            else:
                for value, child in node.children.items():
                    n_child = np.sum(X[:, node.feature] == value)
                    traverse(child, n_child)
        
        traverse(self.tree, len(X))
        
        # Normalisasi
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        return importances
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tree': self.tree,
                'feature_names': self.feature_names,
                'classes': self.classes,
                'params': {
                    'max_depth': self.max_depth,
                    'min_samples_split': self.min_samples_split,
                    'min_samples_leaf': self.min_samples_leaf,
                    'criterion': self.criterion,
                    'random_state': self.random_state,
                    'class_weight': self.class_weight
                }
            }, f)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.tree = data['tree']
        self.feature_names = data['feature_names']
        self.classes = data['classes']
        self.classes_ = self.classes
        params = data['params']
        self.max_depth = params['max_depth']
        self.min_samples_split = params['min_samples_split']
        self.min_samples_leaf = params['min_samples_leaf']
        self.criterion = params.get('criterion', 'entropy')
        self.random_state = params.get('random_state', None)
        self.class_weight = params.get('class_weight', None)
        print(f"✓ Model loaded from: {filepath}")
        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator - for sklearn compatibility"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'random_state': self.random_state,
            'class_weight': self.class_weight
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator - for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def visualize_tree(self, max_depth=3, save_path=None, figsize=(15, 10)):
        if self.tree is None:
            print("Error: Model not trained yet!")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        positions = {}
        self.calc_positions(self.tree, 0, 0, 1, positions, max_depth)
        
        self.draw_node(ax, self.tree, positions, max_depth)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Tree saved to: {save_path}")
        plt.show()
    
    def calc_positions(self, node, depth, left, right, positions, max_depth):
        if node is None or (max_depth and depth > max_depth):
            return 0
        
        if node.is_leaf or (max_depth and depth == max_depth):
            x = (left + right) / 2
            y = -depth
            positions[id(node)] = (x, y)
            return 1
        
        if node.threshold is not None:
            left_leaves = self.calc_positions(node.left, depth + 1, left, 
                                              (left + right) / 2, positions, max_depth)
            right_leaves = self.calc_positions(node.right, depth + 1, 
                                               (left + right) / 2, right, positions, max_depth)
            total = left_leaves + right_leaves
        else:
            total = 0
            n_children = len(node.children)
            if n_children > 0:
                width = (right - left) / n_children
                for i, child in enumerate(node.children.values()):
                    child_left = left + i * width
                    child_right = left + (i + 1) * width
                    total += self.calc_positions(child, depth + 1, child_left, 
                                                  child_right, positions, max_depth)
        
        x = (left + right) / 2
        y = -depth
        positions[id(node)] = (x, y)
        return max(total, 1)
    
    def draw_node(self, ax, node, positions, max_depth, parent_pos=None, edge_label=""):
        if node is None or id(node) not in positions:
            return
        
        x, y = positions[id(node)]
        current_depth = abs(int(y))
        
        if max_depth and current_depth > max_depth:
            return
        
        if parent_pos:
            ax.plot([parent_pos[0], x], [parent_pos[1], y], 'k-', alpha=0.3, linewidth=1)
            if edge_label:
                mid_x, mid_y = (parent_pos[0] + x) / 2, (parent_pos[1] + y) / 2
                ax.text(mid_x, mid_y, edge_label, fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        if node.is_leaf or (max_depth and current_depth == max_depth):
            label = f"Class:\n{node.value}"
            color = 'lightgreen'
        else:
            feat_name = self.feature_names[node.feature]
            if node.threshold is not None:
                label = f"{feat_name}\n≤ {node.threshold:.2f}"
            else:
                label = f"{feat_name}"
            color = 'lightblue'
        
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, 
                        edgecolor='black', linewidth=1.5))
        
        if not node.is_leaf and (not max_depth or current_depth < max_depth):
            if node.threshold is not None:
                if node.left:
                    self.draw_node(ax, node.left, positions, max_depth, (x, y), "≤")
                if node.right:
                    self.draw_node(ax, node.right, positions, max_depth, (x, y), ">")
            else:
                for value, child in node.children.items():
                    self.draw_node(ax, child, positions, max_depth, (x, y), str(value))
    
    def print_tree(self, max_depth=None):
        if self.tree is None:
            print("Model not trained!")
            return
        self.print_node(self.tree, "", True, 0, max_depth)
    
    def print_node(self, node, prefix, is_last, depth, max_depth):
        if node is None or (max_depth and depth > max_depth):
            return
        
        connector = "└── " if is_last else "├── "
        
        if node.is_leaf or (max_depth and depth == max_depth):
            print(f"{prefix}{connector}Class: {node.value}")
        else:
            feat_name = self.feature_names[node.feature]
            if node.threshold is not None:
                print(f"{prefix}{connector}{feat_name} ≤ {node.threshold:.2f}")
            else:
                print(f"{prefix}{connector}{feat_name}")
        
        if not node.is_leaf and (not max_depth or depth < max_depth):
            extension = "    " if is_last else "│   "
            
            if node.threshold is not None:
                if node.left:
                    self.print_node(node.left, prefix + extension, False, depth + 1, max_depth)
                if node.right:
                    self.print_node(node.right, prefix + extension, True, depth + 1, max_depth)
            else:
                children = list(node.children.items())
                for i, (value, child) in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    print(f"{prefix}{extension}{'└── ' if is_last_child else '├── '}[{value}]")
                    self.print_node(child, prefix + extension + 
                                   ("    " if is_last_child else "│   "), 
                                   True, depth + 1, max_depth)
