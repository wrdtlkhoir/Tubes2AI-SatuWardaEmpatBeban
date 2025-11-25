import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.decision_tree_learning import C45DecisionTree


def load_and_explore_data():
    """Load dan explore dataset"""
    print("="*70)
    print("1. LOADING DATA")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv('../data/train.csv')
    
    print(f"\nDataset shape: {train_df.shape}")
    print(f"Number of features: {train_df.shape[1] - 1}")  # -1 untuk target
    print(f"Number of samples: {train_df.shape[0]}")
    
    # Check target distribution
    print(f"\nTarget distribution:")
    print(train_df['Target'].value_counts())
    print(f"\nTarget distribution (%):")
    print(train_df['Target'].value_counts(normalize=True) * 100)
    
    # Check missing values
    print(f"\nMissing values per column:")
    missing = train_df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    # Basic statistics
    print(f"\nDataset info:")
    print(train_df.info())
    
    return train_df


def prepare_data(train_df):
    """Prepare data untuk training"""
    print("\n" + "="*70)
    print("2. PREPARING DATA")
    print("="*70)
    
    # Drop Student_ID (tidak berguna untuk klasifikasi)
    if 'Student_ID' in train_df.columns:
        print("\nDropping Student_ID column (not useful for prediction)...")
        train_df = train_df.drop('Student_ID', axis=1)
    
    # Separate features and target
    X = train_df.drop('Target', axis=1)
    y = train_df['Target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    print(f"\nTraining set target distribution:")
    print(y_train.value_counts())
    
    return X_train, X_val, y_train, y_val


def train_model_with_configs(X_train, y_train, X_val, y_val):
    """Train model dengan berbagai konfigurasi"""
    print("\n" + "="*70)
    print("3. TRAINING MODELS WITH DIFFERENT CONFIGURATIONS")
    print("="*70)
    
    configs = [
        {
            'name': 'Default',
            'params': {}
        },
        {
            'name': 'Shallow Tree (max_depth=5)',
            'params': {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5}
        },
        {
            'name': 'Balanced (max_depth=10)',
            'params': {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'min_gain_ratio': 0.01}
        },
        {
            'name': 'Deep Tree (max_depth=15)',
            'params': {'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'min_gain_ratio': 0.01}
        },
        {
            'name': 'Conservative (prevent overfitting)',
            'params': {'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 10, 'min_gain_ratio': 0.05}
        }
    ]
    
    results = []
    models = []
    
    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Training: {config['name']}")
        print(f"Parameters: {config['params']}")
        
        # Train model
        model = C45DecisionTree(**config['params'])
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        depth = model.get_depth()
        n_leaves = model.get_n_leaves()
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Tree Depth: {depth}")
        print(f"Number of Leaves: {n_leaves}")
        
        # Check overfitting
        overfitting = train_acc - val_acc
        if overfitting > 0.1:
            print(f"⚠️  Warning: Possible overfitting (gap: {overfitting:.4f})")
        elif overfitting < -0.05:
            print(f"⚠️  Warning: Possible underfitting")
        else:
            print(f"✓ Good generalization (gap: {overfitting:.4f})")
        
        results.append({
            'Configuration': config['name'],
            'Train Acc': f"{train_acc:.4f}",
            'Val Acc': f"{val_acc:.4f}",
            'Depth': depth,
            'Leaves': n_leaves,
            'Overfitting': f"{overfitting:.4f}"
        })
        
        models.append({
            'name': config['name'],
            'model': model,
            'val_acc': val_acc
        })
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY - Model Comparison")
    print("="*70)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model_info = max(models, key=lambda x: x['val_acc'])
    print(f"\n✓ Best model: {best_model_info['name']}")
    print(f"  Validation Accuracy: {best_model_info['val_acc']:.4f}")
    
    return best_model_info['model'], models


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate model secara detail"""
    print("\n" + "="*70)
    print("4. DETAILED EVALUATION")
    print("="*70)
    
    # Predictions
    y_val_pred = model.predict(X_val)
    
    # Classification Report
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(10, 8))
    
    # Get unique classes
    classes = sorted(y_val.unique())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - C4.5 Decision Tree\n(Student Dropout Prediction)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(classes):
        class_mask = y_val == class_name
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_val[class_mask], y_val_pred[class_mask])
            print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")
    
    # Feature types detected
    print("\n" + "="*70)
    print("Feature Types Detected by C4.5:")
    print("="*70)
    
    continuous_features = []
    categorical_features = []
    
    for idx, ftype in model.feature_types_.items():
        feature_name = model.feature_names_[idx]
        if ftype == 'continuous':
            continuous_features.append(feature_name)
        else:
            categorical_features.append(feature_name)
    
    print(f"\nContinuous features ({len(continuous_features)}):")
    for i, fname in enumerate(continuous_features, 1):
        print(f"  {i}. {fname}")
    
    print(f"\nCategorical features ({len(categorical_features)}):")
    for i, fname in enumerate(categorical_features, 1):
        print(f"  {i}. {fname}")


def visualize_tree(model, max_depth=3):
    """Visualize decision tree"""
    print("\n" + "="*70)
    print(f"5. TREE VISUALIZATION (Top-{max_depth} Levels)")
    print("="*70)
    
    # Visual representation
    print(f"\nGenerating tree visualization (max_depth={max_depth})...")
    model.visualize_tree(
        max_depth=max_depth,
        save_path=f'student_dropout_tree_top{max_depth}.png',
        figsize=(25, 15),
        dpi=150
    )
    print(f"✓ Tree visualization saved as 'student_dropout_tree_top{max_depth}.png'")
    
    # Text representation
    print(f"\nTree Structure (Text - Top {max_depth} Levels):")
    print("-"*70)
    text_tree = model.export_text(max_depth=max_depth)
    print(text_tree)


def save_model(model, filename='student_dropout_c45_model.pkl'):
    """Save trained model"""
    print("\n" + "="*70)
    print("6. SAVING MODEL")
    print("="*70)
    
    model.save_model(filename)
    print(f"\n✓ Model saved successfully!")
    
    # Verify by loading
    print("\nVerifying saved model...")
    loaded_model = C45DecisionTree()
    loaded_model.load_model(filename)
    print("✓ Model loaded successfully!")
    
    return filename


def predict_test_set(model, test_csv_path='../data/test.csv'):
    """Predict test set untuk submission"""
    print("\n" + "="*70)
    print("7. PREDICTING TEST SET")
    print("="*70)
    
    if not os.path.exists(test_csv_path):
        print(f"\n⚠️  Test file not found: {test_csv_path}")
        print("Skipping test prediction...")
        return None
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    print(f"\nTest set shape: {test_df.shape}")
    
    # Keep Student_ID for submission
    if 'Student_ID' in test_df.columns:
        student_ids = test_df['Student_ID']
        X_test = test_df.drop('Student_ID', axis=1)
    else:
        student_ids = None
        X_test = test_df
    
    # Drop Target if exists in test set
    if 'Target' in X_test.columns:
        X_test = X_test.drop('Target', axis=1)
    
    # Predict
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Create submission DataFrame
    if student_ids is not None:
        submission = pd.DataFrame({
            'Student_ID': student_ids,
            'Target': predictions
        })
    else:
        submission = pd.DataFrame({
            'Target': predictions
        })
    
    # Save submission
    submission_file = 'submission_c45.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\n✓ Predictions saved to: {submission_file}")
    
    # Show prediction distribution
    print("\nPrediction distribution:")
    print(submission['Target'].value_counts())
    
    return submission_file


def main():
    print("\n" + "="*70)
    print("2. PREPROCESSING DATA")
    print("="*70)
    
    # Load data
    df = pd.read_csv('../data/train.csv')
    
    # Drop kolom ID jika ada
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)
    # Rename kolom jika ada karakter aneh
    if 'Daytime/evening attendance\t' in df.columns:
        df = df.rename(columns={'Daytime/evening attendance\t': 'Daytime/evening attendance'})
    
    # Pisahkan fitur dan target
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    print(f"Target train: {y_train.value_counts().to_dict()}")
    print(f"Target val: {y_val.value_counts().to_dict()}")
    
    print("\n" + "="*70)
    print("3. TRAINING C4.5 DECISION TREE")
    print("="*70)
    
    # Inisialisasi model
    model = C45DecisionTree(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        min_gain_ratio=0.01,
        pruning=True,
        pruning_confidence=0.25
    )
    
    # Training
    model.fit(X_train, y_train)
    print("✓ Model trained!")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    
    print("\n" + "="*70)
    print("4. EVALUASI MODEL")
    print("="*70)
    
    # Training set
    y_train_pred = model.predict(X_train)
    print("Akurasi Training:", accuracy_score(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    
    # Validation set
    y_val_pred = model.predict(X_val)
    print("Akurasi Validasi:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Validation Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_val.png')
    print("Confusion matrix saved as confusion_matrix_val.png")
    plt.close()
    
    print("\n" + "="*70)
    print("5. VISUALISASI TREE (TOP 3 LEVEL)")
    print("="*70)
    model.visualize_tree(max_depth=3, save_path='tree_top3.png')
    print("Tree visualisasi saved as tree_top3.png")
    print("\nStruktur tree (top 3 level):")
    print(model.export_text(max_depth=3))
    
    print("\n" + "="*70)
    print("6. SAVE MODEL")
    print("="*70)
    model.save_model('c45_model_from_train.pkl')
    print("Model telah disimpan ke c45_model_from_train.pkl")

if __name__ == "__main__":
    main()
