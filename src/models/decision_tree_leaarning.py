import numpy as np
import pandas as pd
from collections import Counter
import json
import pickle
from typing import Union, List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


class Node:
    """
    Kelas untuk merepresentasikan node dalam decision tree.
    
    Attributes:
        feature: Nama fitur untuk split di node ini (None untuk leaf node)
        threshold: Threshold untuk continuous features (None untuk categorical)
        value: Class prediction untuk leaf node (None untuk internal node)
        children: Dictionary dari child nodes (untuk categorical splits)
        left: Left child untuk continuous splits (threshold <=)
        right: Right child untuk continuous splits (threshold >)
        is_leaf: Boolean apakah node ini adalah leaf
        class_distribution: Distribusi kelas di node ini
        samples: Jumlah sampel di node ini
        depth: Kedalaman node dari root
    """
    
    def __init__(self, feature=None, threshold=None, value=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.children = {}  # For categorical features
        self.left = None    # For continuous features
        self.right = None   # For continuous features
        self.is_leaf = is_leaf
        self.class_distribution = {}
        self.samples = 0
        self.depth = 0
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(value={self.value}, samples={self.samples})"
        elif self.threshold is not None:
            return f"Node(feature={self.feature}, threshold={self.threshold:.2f}, samples={self.samples})"
        else:
            return f"Node(feature={self.feature}, samples={self.samples})"


class C45DecisionTree:
    """
    C4.5 Decision Tree Classifier
    
    Parameters:
        max_depth: Maximum depth of the tree (default: None, unlimited)
        min_samples_split: Minimum samples required to split a node (default: 2)
        min_samples_leaf: Minimum samples required at a leaf node (default: 1)
        min_gain_ratio: Minimum gain ratio required for a split (default: 0.0)
        pruning: Whether to apply post-pruning (default: True)
        pruning_confidence: Confidence level for pruning (default: 0.25)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain_ratio: float = 0.0,
        pruning: bool = True,
        pruning_confidence: float = 0.25
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.pruning = pruning
        self.pruning_confidence = pruning_confidence
        
        self.tree_ = None
        self.feature_names_ = None
        self.classes_ = None
        self.n_features_ = None
        self.n_classes_ = None
        self.feature_types_ = {}  # 'continuous' or 'categorical'
        
    def entropy(self, y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
        Menghitung entropy dari label target.
        
        H(S) = -Σ(p_i * log2(p_i))
        """
        if len(y) == 0:
            return 0.0
        
        if weights is None:
            weights = np.ones(len(y))
        
        # Hitung weighted class distribution
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0.0
        
        # Count weighted occurrences
        unique_classes = np.unique(y)
        entropy = 0.0
        
        for cls in unique_classes:
            mask = (y == cls)
            p = np.sum(weights[mask]) / total_weight
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def gain(self, X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: Optional[float] = None, weights: Optional[np.ndarray] = None) -> float:
        """
        Menghitung Information Gain untuk suatu feature.
        
        IG(S, A) = H(S) - Σ(|S_v|/|S| * H(S_v))
        """
        if weights is None:
            weights = np.ones(len(y))
        
        # Parent entropy
        parent_entropy = self._entropy(y, weights)
        
        feature_values = X[:, feature_idx]
        total_weight = np.sum(weights)
        
        # Handle missing values
        missing_mask = pd.isna(feature_values)
        non_missing_mask = ~missing_mask
        
        if np.all(missing_mask):
            return 0.0
        
        weighted_entropy = 0.0
        
        # Continuous feature dengan threshold
        if threshold is not None:
            non_missing_values = feature_values[non_missing_mask]
            non_missing_y = y[non_missing_mask]
            non_missing_weights = weights[non_missing_mask]
            
            left_mask = non_missing_values <= threshold
            right_mask = non_missing_values > threshold
            
            # Left branch
            if np.any(left_mask):
                left_weight = np.sum(non_missing_weights[left_mask])
                left_entropy = self._entropy(non_missing_y[left_mask], non_missing_weights[left_mask])
                weighted_entropy += (left_weight / total_weight) * left_entropy
            
            # Right branch
            if np.any(right_mask):
                right_weight = np.sum(non_missing_weights[right_mask])
                right_entropy = self._entropy(non_missing_y[right_mask], non_missing_weights[right_mask])
                weighted_entropy += (right_weight / total_weight) * right_entropy
        
        # Categorical feature
        else:
            non_missing_values = feature_values[non_missing_mask]
            non_missing_y = y[non_missing_mask]
            non_missing_weights = weights[non_missing_mask]
            
            unique_values = np.unique(non_missing_values[~pd.isna(non_missing_values)])
            
            for value in unique_values:
                value_mask = (non_missing_values == value)
                if np.any(value_mask):
                    value_weight = np.sum(non_missing_weights[value_mask])
                    value_entropy = self._entropy(non_missing_y[value_mask], non_missing_weights[value_mask])
                    weighted_entropy += (value_weight / total_weight) * value_entropy
        
        return parent_entropy - weighted_entropy
    
    def split_information(self, X: np.ndarray, feature_idx: int, threshold: Optional[float] = None, weights: Optional[np.ndarray] = None) -> float:
        """
        Menghitung Split Information untuk gain ratio.
        
        SI(S, A) = -Σ(|S_v|/|S| * log2(|S_v|/|S|))
        """
        if weights is None:
            weights = np.ones(len(X))
        
        feature_values = X[:, feature_idx]
        total_weight = np.sum(weights)
        
        # Handle missing values
        missing_mask = pd.isna(feature_values)
        non_missing_mask = ~missing_mask
        
        if np.all(missing_mask):
            return 0.0
        
        split_info = 0.0
        
        # Continuous feature
        if threshold is not None:
            non_missing_values = feature_values[non_missing_mask]
            non_missing_weights = weights[non_missing_mask]
            
            left_mask = non_missing_values <= threshold
            right_mask = non_missing_values > threshold
            
            # Left branch
            if np.any(left_mask):
                left_weight = np.sum(non_missing_weights[left_mask])
                p_left = left_weight / total_weight
                if p_left > 0:
                    split_info -= p_left * np.log2(p_left)
            
            # Right branch
            if np.any(right_mask):
                right_weight = np.sum(non_missing_weights[right_mask])
                p_right = right_weight / total_weight
                if p_right > 0:
                    split_info -= p_right * np.log2(p_right)
        
        # Categorical feature
        else:
            non_missing_values = feature_values[non_missing_mask]
            non_missing_weights = weights[non_missing_mask]
            
            unique_values = np.unique(non_missing_values[~pd.isna(non_missing_values)])
            
            for value in unique_values:
                value_mask = (non_missing_values == value)
                if np.any(value_mask):
                    value_weight = np.sum(non_missing_weights[value_mask])
                    p_value = value_weight / total_weight
                    if p_value > 0:
                        split_info -= p_value * np.log2(p_value)
        
        return split_info
    
    def gain_ratio(self, X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: Optional[float] = None, weights: Optional[np.ndarray] = None) -> float:
        """
        Menghitung Gain Ratio (C4.5 improvement over ID3).
        
        GR(S, A) = G(S, A) / SI(S, A)
        """
        gain = self.gain(X, y, feature_idx, threshold, weights)
        split_info = self.split_information(X, feature_idx, threshold, weights)
        
        # Avoid division by zero
        if split_info == 0 or split_info < 1e-10:
            return 0.0
        
        return gain / split_info
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray, available_features: List[int], weights: Optional[np.ndarray] = None) -> Tuple[Optional[int], Optional[float], float]:
        best_gain_ratio = -float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in available_features:
            feature_values = X[:, feature_idx]
            
            # Skip jika semua values missing
            if np.all(pd.isna(feature_values)):
                continue
            
            # Determine feature type
            feature_type = self.feature_types_.get(feature_idx, 'categorical')
            
            if feature_type == 'continuous':
                # Untuk continuous feature, coba beberapa threshold
                non_missing_values = feature_values[~pd.isna(feature_values)]
                
                if len(non_missing_values) < 2:
                    continue
                
                # Sort dan ambil unique values
                sorted_values = np.unique(non_missing_values)
                
                # Coba threshold di antara nilai-nilai berurutan
                # Batasi jumlah threshold untuk efisiensi
                if len(sorted_values) > 10:
                    # Ambil sampel dari sorted values
                    indices = np.linspace(0, len(sorted_values) - 1, 10, dtype=int)
                    sorted_values = sorted_values[indices]
                
                for i in range(len(sorted_values) - 1):
                    threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                    
                    gain_ratio = self._gain_ratio(X, y, feature_idx, threshold, weights)
                    
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feature = feature_idx
                        best_threshold = threshold
            
            else:
                # Categorical feature
                gain_ratio = self._gain_ratio(X, y, feature_idx, None, weights)
                
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
                    best_threshold = None
        
        return best_feature, best_threshold, best_gain_ratio
    
    def majority_class(self, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Any:
        if weights is None:
            weights = np.ones(len(y))
        
        unique_classes = np.unique(y)
        max_weight = -1
        majority = None
        
        for cls in unique_classes:
            mask = (y == cls)
            cls_weight = np.sum(weights[mask])
            if cls_weight > max_weight:
                max_weight = cls_weight
                majority = cls
        
        return majority
    
    def get_class_distribution(self, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict:
        if weights is None:
            weights = np.ones(len(y))
        
        distribution = {}
        unique_classes = np.unique(y)
        
        for cls in unique_classes:
            mask = (y == cls)
            distribution[cls] = np.sum(weights[mask])
        
        return distribution
    
    def build_tree(self, X: np.ndarray, y: np.ndarray, available_features: List[int], depth: int = 0, weights: Optional[np.ndarray] = None) -> Node:
        if weights is None:
            weights = np.ones(len(y))
        
        # Base cases untuk stopping criteria
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Create node
        node = Node()
        node.samples = n_samples
        node.depth = depth
        node.class_distribution = self._get_class_distribution(y, weights)
        
        # Stopping criteria
        # 1. Semua instances termasuk kelas yang sama
        if n_classes == 1:
            node.is_leaf = True
            node.value = y[0]
            return node
        
        # 2. Tidak ada features yang tersisa atau max depth tercapai
        if len(available_features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            node.is_leaf = True
            node.value = self._majority_class(y, weights)
            return node
        
        # 3. Jumlah samples terlalu kecil untuk split
        if n_samples < self.min_samples_split:
            node.is_leaf = True
            node.value = self._majority_class(y, weights)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain_ratio = self._find_best_split(
            X, y, available_features, weights
        )
        
        # 4. Tidak ada split yang memberikan gain yang cukup
        if best_feature is None or best_gain_ratio < self.min_gain_ratio:
            node.is_leaf = True
            node.value = self._majority_class(y, weights)
            return node
        
        # Set node properties
        node.feature = best_feature
        node.threshold = best_threshold
        
        feature_values = X[:, best_feature]
        
        # Continuous feature split
        if best_threshold is not None:
            # Handle missing values: distribute proportionally
            missing_mask = pd.isna(feature_values)
            non_missing_mask = ~missing_mask
            
            if np.any(non_missing_mask):
                non_missing_values = feature_values[non_missing_mask]
                
                left_mask_non_missing = non_missing_values <= best_threshold
                right_mask_non_missing = non_missing_values > best_threshold
                
                # Hitung proporsi untuk missing values
                left_proportion = np.sum(weights[non_missing_mask][left_mask_non_missing]) / np.sum(weights[non_missing_mask])
                right_proportion = 1 - left_proportion
                
                # Create masks untuk semua data
                left_mask = np.zeros(len(y), dtype=bool)
                right_mask = np.zeros(len(y), dtype=bool)
                
                left_mask[non_missing_mask] = left_mask_non_missing
                right_mask[non_missing_mask] = right_mask_non_missing
                
                # Duplicate missing instances ke kedua branches dengan weights yang disesuaikan
                left_weights = weights.copy()
                right_weights = weights.copy()
                
                if np.any(missing_mask):
                    left_mask[missing_mask] = True
                    right_mask[missing_mask] = True
                    left_weights[missing_mask] *= left_proportion
                    right_weights[missing_mask] *= right_proportion
                
                # Build left subtree
                if np.any(left_mask) and np.sum(left_weights[left_mask]) >= self.min_samples_leaf:
                    node.left = self._build_tree(
                        X[left_mask],
                        y[left_mask],
                        available_features,
                        depth + 1,
                        left_weights[left_mask]
                    )
                else:
                    # Create leaf
                    leaf = Node(is_leaf=True)
                    leaf.value = self._majority_class(y, weights)
                    leaf.samples = 0
                    leaf.depth = depth + 1
                    node.left = leaf
                
                # Build right subtree
                if np.any(right_mask) and np.sum(right_weights[right_mask]) >= self.min_samples_leaf:
                    node.right = self._build_tree(
                        X[right_mask],
                        y[right_mask],
                        available_features,
                        depth + 1,
                        right_weights[right_mask]
                    )
                else:
                    # Create leaf
                    leaf = Node(is_leaf=True)
                    leaf.value = self._majority_class(y, weights)
                    leaf.samples = 0
                    leaf.depth = depth + 1
                    node.right = leaf
        
        # Categorical feature split
        else:
            missing_mask = pd.isna(feature_values)
            non_missing_mask = ~missing_mask
            
            if np.any(non_missing_mask):
                non_missing_values = feature_values[non_missing_mask]
                unique_values = np.unique(non_missing_values)
                
                # Hitung proporsi untuk setiap value (untuk missing value distribution)
                value_proportions = {}
                total_non_missing_weight = np.sum(weights[non_missing_mask])
                
                for value in unique_values:
                    value_mask = (non_missing_values == value)
                    value_proportions[value] = np.sum(weights[non_missing_mask][value_mask]) / total_non_missing_weight
                
                # Categorical features tidak dihapus dari available features untuk C4.5
                # Namun untuk simplifikasi, kita hapus (bisa diubah sesuai kebutuhan)
                new_available_features = [f for f in available_features if f != best_feature]
                
                # Build child untuk setiap unique value
                for value in unique_values:
                    value_mask = np.zeros(len(y), dtype=bool)
                    value_mask[non_missing_mask] = (non_missing_values == value)
                    
                    value_weights = weights.copy()
                    
                    # Distribute missing instances ke branch ini dengan adjusted weight
                    if np.any(missing_mask):
                        value_mask[missing_mask] = True
                        value_weights[missing_mask] *= value_proportions[value]
                    
                    if np.any(value_mask) and np.sum(value_weights[value_mask]) >= self.min_samples_leaf:
                        child = self._build_tree(
                            X[value_mask],
                            y[value_mask],
                            new_available_features,
                            depth + 1,
                            value_weights[value_mask]
                        )
                        node.children[value] = child
                    else:
                        # Create leaf
                        leaf = Node(is_leaf=True)
                        leaf.value = self._majority_class(y[value_mask], value_weights[value_mask]) if np.any(value_mask) else self._majority_class(y, weights)
                        leaf.samples = np.sum(value_mask)
                        leaf.depth = depth + 1
                        node.children[value] = leaf
        
        return node
    
    