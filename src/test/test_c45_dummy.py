import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'e:/SEMESTER_5/AI/Tubes2AI-SatuWardaEmpatBeban/src')
from models.decision_tree_learning import C45DecisionTree

np.random.seed(42)

print("=" * 80)
print("TEST: SHOW GAIN VALUES FOR ALL ATTRIBUTES")
print("=" * 80)

X_train = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 35, 28, 40, 33, np.nan, 50],
    'salary': [50000, 60000, 55000, 80000, np.nan, 52000, 75000, 62000, 58000, 90000],
    'dept': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales'],
    'gender': ['M', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M']
})
y_train = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])

print("\nData:")
display_df = X_train.copy()
display_df['Target'] = y_train
print(display_df.to_string())

print("\n" + "=" * 80)
print("CALCULATING INFORMATION GAIN FOR EACH ATTRIBUTE")
print("=" * 80)

model = C45DecisionTree()
X_values = X_train.values

parent_entropy = model.entropy(y_train)
print(f"\nParent Entropy: {parent_entropy:.4f}")
print(f"Target distribution: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")

gains = []

for feature_idx, feature_name in enumerate(X_train.columns):
    print(f"\n{'='*80}")
    print(f"ATTRIBUTE: {feature_name}")
    print(f"{'='*80}")
    
    x_feature = X_values[:, feature_idx]
    
    if pd.isna(x_feature).all():
        print("  ⚠ All values are missing - SKIP")
        continue
    
    is_continuous = model.is_continuous(x_feature)
    print(f"Type: {'CONTINUOUS' if is_continuous else 'CATEGORICAL'}")
    
    if is_continuous:
        mask_valid = ~pd.isna(x_feature)
        x_clean = x_feature[mask_valid]
        y_clean = y_train[mask_valid]
        
        sorted_indices = np.argsort(x_clean)
        x_sorted = x_clean[sorted_indices]
        y_sorted = y_clean[sorted_indices]
        
        print(f"\nSorted values (non-missing):")
        for i, (val, label) in enumerate(zip(x_sorted, y_sorted)):
            val_str = f"{val:.1f}" if isinstance(val, (int, float, np.number)) else str(val)
            print(f"  {i}: {feature_name}={val_str}, Target={label}")
        
        candidate_thresholds = []
        for i in range(len(y_sorted) - 1):
            if y_sorted[i] != y_sorted[i + 1]:
                c = (x_sorted[i] + x_sorted[i + 1]) / 2
                candidate_thresholds.append(c)
                print(f"\n  Label change at index {i}→{i+1}: {x_sorted[i]:.1f}(class {y_sorted[i]}) → {x_sorted[i+1]:.1f}(class {y_sorted[i+1]})")
                print(f"    → Candidate threshold C = {c:.2f}")
        
        if not candidate_thresholds:
            print("\n  ⚠ No label changes found - SKIP")
            continue
        
        best_threshold = None
        best_gain = -1
        
        print(f"\nEvaluating {len(candidate_thresholds)} candidate threshold(s):")
        for threshold in candidate_thresholds:
            gain = model.information_gain(X_values, y_train, feature_idx, threshold)
            print(f"  C = {threshold:.2f}: Gain = {gain:.4f}")
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        print(f"\n  ✓ Best threshold: C = {best_threshold:.2f}")
        print(f"  ✓ Best gain: {best_gain:.4f}")
        
        left_mask = X_values[:, feature_idx] <= best_threshold
        left_mask = left_mask & ~pd.isna(X_values[:, feature_idx])
        right_mask = X_values[:, feature_idx] > best_threshold
        right_mask = right_mask & ~pd.isna(X_values[:, feature_idx])
        
        print(f"\n  Split preview with C = {best_threshold:.2f}:")
        print(f"    {feature_name} ≤ {best_threshold:.2f}: {left_mask.sum()} samples → Class 0={sum(y_train[left_mask]==0)}, Class 1={sum(y_train[left_mask]==1)}")
        print(f"    {feature_name} > {best_threshold:.2f}: {right_mask.sum()} samples → Class 0={sum(y_train[right_mask]==0)}, Class 1={sum(y_train[right_mask]==1)}")
        
        gains.append((feature_name, best_gain, f"C={best_threshold:.2f}"))
        
    else:
        gain = model.information_gain(X_values, y_train, feature_idx, None)
        
        mask_valid = ~pd.isna(x_feature)
        x_clean = x_feature[mask_valid]
        values = np.unique(x_clean)
        
        print(f"\nUnique values: {list(values)}")
        print(f"\nSplit preview:")
        for value in values:
            mask = (X_values[:, feature_idx] == value)
            n_samples = mask.sum()
            n_class0 = sum(y_train[mask] == 0)
            n_class1 = sum(y_train[mask] == 1)
            print(f"  {feature_name} = {value}: {n_samples} samples → Class 0={n_class0}, Class 1={n_class1}")
        
        print(f"\n  ✓ Gain: {gain:.4f}")
        gains.append((feature_name, gain, "categorical"))

print("\n" + "=" * 80)
print("SUMMARY: INFORMATION GAIN FOR ALL ATTRIBUTES")
print("=" * 80)

gains.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Attribute':<15} {'Gain':<12} {'Details'}")
print("-" * 80)
for i, (attr, gain, details) in enumerate(gains, 1):
    print(f"{i:<6} {attr:<15} {gain:<12.4f} {details}")

print(f"\n✓ BEST ATTRIBUTE TO SPLIT: {gains[0][0]} (Gain = {gains[0][1]:.4f})")

print("\n" + "=" * 80)
print("BUILDING TREE WITH SELECTED ATTRIBUTE")
print("=" * 80)

model = C45DecisionTree(max_depth=3, min_samples_split=2)
model.fit(X_train, y_train)

print("\nDecision Tree Structure:")
model.print_tree()

train_pred = model.predict(X_train)
train_acc = (train_pred == y_train).mean()
print(f"\n✓ Train Accuracy: {train_acc:.2%}")
