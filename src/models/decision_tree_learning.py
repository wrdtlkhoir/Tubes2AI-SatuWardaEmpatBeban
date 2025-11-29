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
        
        
class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
        self.feature_names = None
        self.classes = None
        
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"X{i}" for i in range(X.shape[1])]
            
        if isinstance(y, pd.Series):
            y = y.values
            
        self.classes = np.unique(y)
        self.tree = self.build_tree(X, y, depth=0)
        return self
    
    def is_continuous(self, x):
        try:
            x_clean = x[~pd.isna(x)]
            if len(x_clean) == 0:
                return False
            x_clean.astype(float)
            return len(np.unique(x_clean)) > 10
        except:
            return False
    
    def entropy(self, y):
        if len(y) == 0:
            return 0
        props = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in props if p > 0])
    
    def information_gain(self, X, y, feature_idx, threshold=None):
        parent_entropy = self.entropy(y)
        
        mask = ~pd.isna(X[:, feature_idx])
        if mask.sum() == 0:
            return 0
        
        X_valid = X[mask]
        y_valid = y[mask]
        
        if threshold is not None:
            left_mask = X_valid[:, feature_idx] <= threshold
            right_mask = ~left_mask
            splits = [(y_valid[left_mask], len(y_valid[left_mask])), 
                     (y_valid[right_mask], len(y_valid[right_mask]))]
        else:
            values = np.unique(X_valid[:, feature_idx])
            splits = [(y_valid[X_valid[:, feature_idx] == v], 
                      len(y_valid[X_valid[:, feature_idx] == v])) for v in values]
        
        n_total = len(y_valid)
        weighted_entropy = 0
        
        for split_y, n_split in splits:
            if n_split > 0:
                weight = n_split / n_total
                weighted_entropy += weight * self.entropy(split_y)
        
        gain = parent_entropy - weighted_entropy
        return gain
    
    def gain_ratio(self, X, y, feature_idx, threshold=None):
        gain = self.information_gain(X, y, feature_idx, threshold)
        
        mask = ~pd.isna(X[:, feature_idx])
        if mask.sum() == 0:
            return 0
        
        X_valid = X[mask]
        
        if threshold is not None:
            left_mask = X_valid[:, feature_idx] <= threshold
            right_mask = ~left_mask
            splits = [left_mask.sum(), right_mask.sum()]
        else:
            values = np.unique(X_valid[:, feature_idx])
            splits = [(X_valid[:, feature_idx] == v).sum() for v in values]
        
        n_total = len(X_valid)
        split_info = 0
        
        for n_split in splits:
            if n_split > 0:
                weight = n_split / n_total
                split_info -= weight * np.log2(weight)
        
        if split_info == 0:
            return 0
        return gain / split_info
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            x_feature = X[:, feature_idx]
            
            if pd.isna(x_feature).all():
                continue
            
            if self.is_continuous(x_feature):
                x_clean = x_feature[~pd.isna(x_feature)]
                thresholds = np.percentile(x_clean, [25, 50, 75])
                
                for threshold in thresholds:
                    if self.criterion == 'gain_ratio':
                        gain = self.gain_ratio(X, y, feature_idx, threshold)
                    else:
                        gain = self.information_gain(X, y, feature_idx, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            else:
                if self.criterion == 'gain_ratio':
                    gain = self.gain_ratio(X, y, feature_idx, None)
                else:
                    gain = self.information_gain(X, y, feature_idx, None)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = None
        
        return best_feature, best_threshold
    
    def majority_class(self, y):
        return np.bincount(y).argmax()
    
    def build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        node = Node()
        
        if (n_classes == 1 or 
            n_samples < self.min_samples_split or
            (self.max_depth and depth >= self.max_depth)):
            node.is_leaf = True
            node.value = self.classes[self.majority_class(y)]
            return node
        
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            node.is_leaf = True
            node.value = self.classes[self.majority_class(y)]
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        
        x_feature = X[:, best_feature]
        
        if best_threshold is not None:
            mask_valid = ~pd.isna(x_feature)
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
                node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
            if right_mask.sum() >= self.min_samples_leaf:
                node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
                
        else:
            values = np.unique(x_feature[~pd.isna(x_feature)])
            for value in values:
                mask = x_feature == value
                if mask.sum() >= self.min_samples_leaf:
                    node.children[value] = self.build_tree(X[mask], y[mask], depth + 1)
        
        return node
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self.predict_sample(x, self.tree) for x in X])
    
    def predict_sample(self, x, node):
        if node.is_leaf:
            return node.value
        
        feature_val = x[node.feature]
        
        if pd.isna(feature_val):
            if node.left:
                return self.predict_sample(x, node.left)
            elif node.right:
                return self.predict_sample(x, node.right)
            elif node.children:
                return self.predict_sample(x, list(node.children.values())[0])
            return self.classes[0]
        
        if node.threshold is not None:
            if feature_val <= node.threshold:
                return self.predict_sample(x, node.left) if node.left else self.classes[0]
            else:
                return self.predict_sample(x, node.right) if node.right else self.classes[0]
        else:
            if feature_val in node.children:
                return self.predict_sample(x, node.children[feature_val])
            return self.classes[0]
    
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
                    'criterion': self.criterion
                }
            }, f)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.tree = data['tree']
        self.feature_names = data['feature_names']
        self.classes = data['classes']
        params = data['params']
        self.max_depth = params['max_depth']
        self.min_samples_split = params['min_samples_split']
        self.min_samples_leaf = params['min_samples_leaf']
        self.criterion = params.get('criterion', 'entropy')
        print(f"✓ Model loaded from: {filepath}")
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
