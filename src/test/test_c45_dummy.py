import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'e:/SEMESTER_5/AI/Tubes2AI-SatuWardaEmpatBeban/src')
from models.decision_tree_learning import C45DecisionTree

np.random.seed(42)

print("=" * 60)
print("TEST C4.5 DECISION TREE - DATA DUMMY")
print("=" * 60)

print("\n1. DATA CAMPURAN (Numeric + Categorical + Missing)")
print("-" * 60)

X_train = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 35, 28, 40, 33, np.nan, 50],
    'salary': [50000, 60000, 55000, 80000, np.nan, 52000, 75000, 62000, 58000, 90000],
    'dept': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales'],
    'gender': ['M', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M']
})
y_train = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])

X_test = pd.DataFrame({
    'age': [27, 42, 31, np.nan],
    'salary': [53000, 78000, np.nan, 85000],
    'dept': ['IT', 'Sales', 'HR', 'Sales'],
    'gender': ['M', 'M', 'F', 'F']
})
y_test = np.array([0, 1, 0, 1])

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Null values train: {X_train.isnull().sum().sum()}")
print(f"Null values test: {X_test.isnull().sum().sum()}")

model = C45DecisionTree(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = (train_pred == y_train).mean()
test_acc = (test_pred == y_test).mean()

print(f"\n✓ Train Accuracy: {train_acc:.2%}")
print(f"✓ Test Accuracy: {test_acc:.2%}")

print("\n2. STRUKTUR TREE (Top 3 levels)")
print("-" * 60)
model.print_tree(max_depth=3)

print("\n3. SAVE & LOAD MODEL")
print("-" * 60)
model.save_model('test_c45.pkl')
loaded = C45DecisionTree().load_model('test_c45.pkl')
loaded_pred = loaded.predict(X_test)
print(f"✓ Loaded model accuracy: {(loaded_pred == y_test).mean():.2%}")

print("\n4. VISUALISASI")
print("-" * 60)
print("Generating tree visualization...")
model.visualize_tree(max_depth=3, save_path='tree_dummy.png', figsize=(12, 8))

print("\n" + "=" * 60)
print("TEST SELESAI!")
print("=" * 60)
