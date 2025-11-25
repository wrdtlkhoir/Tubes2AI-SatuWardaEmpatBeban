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
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'C45DecisionTree':
        """
        Fit decision tree classifier.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]
        if isinstance(y, pd.Series):
            y = y.values
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # Detect feature types (continuous or categorical)
        self.feature_types_ = {}
        for i in range(self.n_features_):
            feature_values = X[:, i]
            non_missing = feature_values[~pd.isna(feature_values)]
            if len(non_missing) == 0:
                self.feature_types_[i] = 'categorical'
                continue
            try:
                _ = non_missing.astype(float)
                self.feature_types_[i] = 'continuous'
            except:
                self.feature_types_[i] = 'categorical'
        available_features = list(range(self.n_features_))
        self.tree_ = self.build_tree(X, y, available_features, depth=0)
        return self
    
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
        parent_entropy = self.entropy(y, weights)
        
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
                left_entropy = self.entropy(non_missing_y[left_mask], non_missing_weights[left_mask])
                weighted_entropy += (left_weight / total_weight) * left_entropy
            
            # Right branch
            if np.any(right_mask):
                right_weight = np.sum(non_missing_weights[right_mask])
                right_entropy = self.entropy(non_missing_y[right_mask], non_missing_weights[right_mask])
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
                    value_entropy = self.entropy(non_missing_y[value_mask], non_missing_weights[value_mask])
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
                    
                    gain_ratio = self.gain_ratio(X, y, feature_idx, threshold, weights)
                    
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feature = feature_idx
                        best_threshold = threshold
            
            else:
                # Categorical feature
                gain_ratio = self.gain_ratio(X, y, feature_idx, None, weights)
                
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
        node.class_distribution = self.get_class_distribution(y, weights)
        
        # Stopping criteria
        # 1. Semua instances termasuk kelas yang sama
        if n_classes == 1:
            node.is_leaf = True
            node.value = y[0]
            return node
        
        # 2. Tidak ada features yang tersisa atau max depth tercapai
        if len(available_features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            node.is_leaf = True
            node.value = self.majority_class(y, weights)
            return node
        
        # 3. Jumlah samples terlalu kecil untuk split
        if n_samples < self.min_samples_split:
            node.is_leaf = True
            node.value = self.majority_class(y, weights)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain_ratio = self.find_best_split(
            X, y, available_features, weights
        )
        
        # 4. Tidak ada split yang memberikan gain yang cukup
        if best_feature is None or best_gain_ratio < self.min_gain_ratio:
            node.is_leaf = True
            node.value = self.majority_class(y, weights)
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
                    node.left = self.build_tree(
                        X[left_mask],
                        y[left_mask],
                        available_features,
                        depth + 1,
                        left_weights[left_mask]
                    )
                else:
                    # Create leaf
                    leaf = Node(is_leaf=True)
                    leaf.value = self.majority_class(y, weights)
                    leaf.samples = 0
                    leaf.depth = depth + 1
                    node.left = leaf
                
                # Build right subtree
                if np.any(right_mask) and np.sum(right_weights[right_mask]) >= self.min_samples_leaf:
                    node.right = self.build_tree(
                        X[right_mask],
                        y[right_mask],
                        available_features,
                        depth + 1,
                        right_weights[right_mask]
                    )
                else:
                    # Create leaf
                    leaf = Node(is_leaf=True)
                    leaf.value = self.majority_class(y, weights)
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
                        child = self.build_tree(
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
                        leaf.value = self.majority_class(y[value_mask], value_weights[value_mask]) if np.any(value_mask) else self.majority_class(y, weights)
                        leaf.samples = np.sum(value_mask)
                        leaf.depth = depth + 1
                        node.children[value] = leaf
        
        return node
    
    def save_model(self, filepath: str) -> None:
        model_data = {
            'tree': self.tree_,
            'feature_names': self.feature_names_,
            'classes': self.classes_,
            'n_features': self.n_features_,
            'n_classes': self.n_classes_,
            'feature_types': self.feature_types_,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_gain_ratio': self.min_gain_ratio,
            'pruning': self.pruning,
            'pruning_confidence': self.pruning_confidence
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model berhasil disimpan ke: {filepath}")
    
    def load_model(self, filepath: str) -> 'C45DecisionTree':
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tree_ = model_data['tree']
        self.feature_names_ = model_data['feature_names']
        self.classes_ = model_data['classes']
        self.n_features_ = model_data['n_features']
        self.n_classes_ = model_data['n_classes']
        self.feature_types_ = model_data['feature_types']
        self.max_depth = model_data['max_depth']
        self.min_samples_split = model_data['min_samples_split']
        self.min_samples_leaf = model_data['min_samples_leaf']
        self.min_gain_ratio = model_data['min_gain_ratio']
        self.pruning = model_data['pruning']
        self.pruning_confidence = model_data['pruning_confidence']
        
        print(f"Model berhasil dimuat dari: {filepath}")
        return self
    
    def get_depth(self) -> int:
        return self.get_node_depth(self.tree_)
    
    def get_node_depth(self, node: Node) -> int:
        if node is None or node.is_leaf:
            return 0
        
        if node.threshold is not None:
            # Continuous split
            left_depth = self.get_node_depth(node.left) if node.left else 0
            right_depth = self.get_node_depth(node.right) if node.right else 0
            return 1 + max(left_depth, right_depth)
        else:
            # Categorical split
            max_child_depth = 0
            for child in node.children.values():
                child_depth = self.get_node_depth(child)
                max_child_depth = max(max_child_depth, child_depth)
            return 1 + max_child_depth
    
    def get_n_leaves(self) -> int:
        return self.count_leaves(self.tree_)
    
    def count_leaves(self, node: Node) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        
        count = 0
        if node.threshold is not None:
            count += self.count_leaves(node.left)
            count += self.count_leaves(node.right)
        else:
            for child in node.children.values():
                count += self.count_leaves(child)
        
        return count
    
    def visualize_tree(self, max_depth: Optional[int] = None, save_path: Optional[str] = None, figsize: Tuple[int, int] = (20, 12), dpi: int = 100) -> None:
        if self.tree_ is None:
            raise ValueError("Model belum di-fit. Panggil fit() terlebih dahulu.")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis('off')
        
        # Hitung posisi nodes
        node_positions = {}
        self.calculate_positions(self.tree_, 0, 0, 1, node_positions, max_depth)
        
        # Draw tree
        self.draw_tree(ax, self.tree_, node_positions, max_depth)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Visualisasi tree disimpan ke: {save_path}")
        
        plt.show()
    
    def calculate_positions(self, node: Node, depth: int, left: float, right: float, positions: Dict, max_depth: Optional[int]) -> int:
        if node is None or (max_depth is not None and depth > max_depth):
            return 0
        
        if node.is_leaf or (max_depth is not None and depth == max_depth):
            x = (left + right) / 2
            y = -depth
            positions[id(node)] = (x, y)
            return 1
        
        # Recursive untuk children
        if node.threshold is not None:
            # Continuous split
            left_leaves = self.calculate_positions(node.left, depth + 1, left, (left + right) / 2, positions, max_depth)
            right_leaves = self.calculate_positions(node.right, depth + 1, (left + right) / 2, right, positions, max_depth)
            total_leaves = left_leaves + right_leaves
        else:
            # Categorical split
            n_children = len(node.children)
            if n_children == 0:
                x = (left + right) / 2
                y = -depth
                positions[id(node)] = (x, y)
                return 1
            
            width = (right - left) / n_children
            total_leaves = 0
            
            for i, child in enumerate(node.children.values()):
                child_left = left + i * width
                child_right = left + (i + 1) * width
                child_leaves = self.calculate_positions(child, depth + 1, child_left, child_right, positions, max_depth)
                total_leaves += child_leaves
        
        # Position untuk node ini
        x = (left + right) / 2
        y = -depth
        positions[id(node)] = (x, y)
        
        return total_leaves
    
    def draw_tree(self, ax, node: Node, positions: Dict, max_depth: Optional[int], parent_pos: Optional[Tuple[float, float]] = None, edge_label: str = "") -> None:
        if node is None or id(node) not in positions:
            return
        
        current_depth = abs(int(positions[id(node)][1]))
        if max_depth is not None and current_depth > max_depth:
            return
        
        x, y = positions[id(node)]
        
        # Draw edge dari parent
        if parent_pos is not None:
            ax.plot([parent_pos[0], x], [parent_pos[1], y], 'k-', alpha=0.3, linewidth=1.5)
            
            # Edge label
            if edge_label:
                mid_x = (parent_pos[0] + x) / 2
                mid_y = (parent_pos[1] + y) / 2
                ax.text(mid_x, mid_y, edge_label, fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Node label
        if node.is_leaf or (max_depth is not None and current_depth == max_depth):
            # Leaf node
            label = f"Class: {node.value}\nSamples: {node.samples}"
            box_color = 'lightgreen'
        else:
            # Internal node
            feature_name = self.feature_names_[node.feature] if self.feature_names_ else f"F{node.feature}"
            if node.threshold is not None:
                label = f"{feature_name}\n<= {node.threshold:.2f}\nSamples: {node.samples}"
            else:
                label = f"{feature_name}\nSamples: {node.samples}"
            box_color = 'lightblue'
        
        # Draw node
        bbox = dict(boxstyle='round,pad=0.5', facecolor=box_color, edgecolor='black', linewidth=1.5)
        ax.text(x, y, label, fontsize=9, ha='center', va='center', bbox=bbox)
        
        # Draw children
        if not node.is_leaf and (max_depth is None or current_depth < max_depth):
            if node.threshold is not None:
                # Continuous split
                if node.left:
                    self.draw_tree(ax, node.left, positions, max_depth, (x, y), "True")
                if node.right:
                    self.draw_tree(ax, node.right, positions, max_depth, (x, y), "False")
            else:
                # Categorical split
                for value, child in node.children.items():
                    self.draw_tree(ax, child, positions, max_depth, (x, y), str(value))
    
    def export_text(self, max_depth: Optional[int] = None) -> str:
        if self.tree_ is None:
            return "Model belum di-fit."
        
        lines = []
        self.export_text_recursive(self.tree_, "", True, lines, 0, max_depth)
        return "\n".join(lines)
    
    def export_text_recursive(self, node: Node, prefix: str, is_last: bool, lines: List[str], depth: int, max_depth: Optional[int]) -> None:
        if node is None or (max_depth is not None and depth > max_depth):
            return
        
        # Current node
        connector = "└── " if is_last else "├── "
        
        if node.is_leaf or (max_depth is not None and depth == max_depth):
            lines.append(f"{prefix}{connector}Class: {node.value} (samples: {node.samples})")
        else:
            feature_name = self.feature_names_[node.feature] if self.feature_names_ else f"Feature_{node.feature}"
            if node.threshold is not None:
                lines.append(f"{prefix}{connector}{feature_name} <= {node.threshold:.2f} (samples: {node.samples})")
            else:
                lines.append(f"{prefix}{connector}{feature_name} (samples: {node.samples})")
        
        # Children
        if not node.is_leaf and (max_depth is None or depth < max_depth):
            extension = "    " if is_last else "│   "
            
            if node.threshold is not None:
                # Continuous split
                if node.left:
                    self.export_text_recursive(node.left, prefix + extension, False, lines, depth + 1, max_depth)
                if node.right:
                    self.export_text_recursive(node.right, prefix + extension, True, lines, depth + 1, max_depth)
            else:
                # Categorical split
                children_list = list(node.children.items())
                for i, (value, child) in enumerate(children_list):
                    is_last_child = (i == len(children_list) - 1)
                    lines.append(f"{prefix}{extension}{'└── ' if is_last_child else '├── '}[{value}]")
                    self.export_text_recursive(
                        child,
                        prefix + extension + ("    " if is_last_child else "│   "),
                        True,
                        lines,
                        depth + 1,
                        max_depth
                    )
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for samples in X.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict_single(x, self.tree_) for x in X])

    def _predict_single(self, x, node: Node):
        while not node.is_leaf:
            feature_value = x[node.feature]
            if node.threshold is not None:
                # Continuous feature
                if pd.isna(feature_value):
                    # If missing, return majority class at node
                    return node.value
                if feature_value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                # Categorical feature
                if pd.isna(feature_value) or feature_value not in node.children:
                    return node.value
                node = node.children.get(feature_value, node)
        return node.value