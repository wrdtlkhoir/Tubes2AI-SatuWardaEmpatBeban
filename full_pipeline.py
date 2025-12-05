import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from scipy.optimize import minimize
from scipy.special import logsumexp

# ==========================================
# 1. ROBUST MANUAL LOGISTIC REGRESSION
# ==========================================
class ManualLogisticRegression:
    """
    Implementasi Logistic Regression Manual yang meniru perilaku sklearn.
    Menggunakan L-BFGS-B untuk meminimalkan Multinomial Cross-Entropy Loss + L2 Regularization.
    """
    
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, random_state=None, solver='lbfgs', class_weight=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.solver = solver
        self.class_weight = class_weight
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        
    def _softmax(self, z):
        # Numerically stable softmax using logsumexp
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _loss_grad_lbfgs(self, params, X, Y_onehot, alpha, sample_weights=None):
        n_features = X.shape[1]
        n_classes = Y_onehot.shape[1]
        
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

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        encoder = OneHotEncoder(sparse_output=False, categories=[self.classes_])
        Y_onehot = encoder.fit_transform(y.reshape(-1, 1))
        
        sample_weights = None
        if self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            weights = n_samples / (n_classes * class_counts)
            sample_weights = weights[y]

        alpha = 1.0 / self.C
        initial_params = np.zeros(n_features * n_classes + n_classes)
        
        res = minimize(
            fun=self._loss_grad_lbfgs,
            x0=initial_params,
            args=(X, Y_onehot, alpha, sample_weights),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        self.coef_ = res.x[:n_features * n_classes].reshape(n_features, n_classes).T
        self.intercept_ = res.x[n_features * n_classes:]
        return self

    def predict_proba(self, X):
        Z = X @ self.coef_.T + self.intercept_
        return self._softmax(Z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def get_params(self, deep=True):
        return {'C': self.C, 'max_iter': self.max_iter, 'tol': self.tol, 'random_state': self.random_state, 'solver': self.solver, 'class_weight': self.class_weight}

# ==========================================
# 2. MANUAL RFE
# ==========================================
class ManualRFE:
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
            
            # Re-instantiate to reset weights
            params = self.estimator.get_params()
            model = ManualLogisticRegression(**params)
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

# ==========================================
# 3. PIPELINE & ENGINEERING (UNCHANGED)
# ==========================================
print("Loading Data...")
train_df = pd.read_csv('/kaggle/input/if-3170-tugas-besar-2-student-performance/train.csv')
test_df = pd.read_csv('/kaggle/input/if-3170-tugas-besar-2-student-performance/test.csv')

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

X = train_df.drop('Target', axis=1)
y_raw = train_df['Target']
X_test = test_df.copy()

test_ids = test_df['Student_ID']
X = X.drop('Student_ID', axis=1)
X_test = X_test.drop('Student_ID', axis=1)

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
classes = le.classes_

def group_rare_labels(df, col, threshold=0.015):
    counts = df[col].value_counts(normalize=True)
    rare_labels = counts[counts < threshold].index
    return df[col].replace(rare_labels, 'Other')

def engineering_pipeline(df_train, df_test):
    n_train = len(df_train)
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    cols_to_group = ["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Previous qualification"]
    for col in cols_to_group:
        df[col] = df[col].astype(str)
        df[col] = group_rare_labels(df, col)

    for sem in ['1st', '2nd']:
        enrolled = df[f'Curricular units {sem} sem (enrolled)']
        approved = df[f'Curricular units {sem} sem (approved)']
        evals = df[f'Curricular units {sem} sem (evaluations)']
        grade = df[f'Curricular units {sem} sem (grade)']
        
        safe_enrolled = enrolled.replace(0, 1)
        safe_approved = approved.replace(0, 1)
        
        df[f'{sem}_pass_rate'] = approved / safe_enrolled
        df[f'{sem}_persistence'] = evals / safe_approved
        df[f'{sem}_weighted_score'] = grade * df[f'{sem}_pass_rate']
        df[f'{sem}_is_partial'] = ((approved > 0) & (approved < enrolled)).astype(int)
        df[f'{sem}_is_perfect'] = (approved == enrolled) & (enrolled > 0).astype(int)

    df['Grade_Change'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']
    df['Total_Failed'] = (df['Curricular units 1st sem (enrolled)'] - df['Curricular units 1st sem (approved)']) + (df['Curricular units 2nd sem (enrolled)'] - df['Curricular units 2nd sem (approved)'])
    df['Financial_Crisis'] = (df['Debtor'] * 2) + (1 - df['Tuition fees up to date'])
    df['Age_Log'] = np.log1p(df['Age at enrollment'])
    
    return df.iloc[:n_train], df.iloc[n_train:]

print("Menerapkan Engineering...")
X_eng, X_test_eng = engineering_pipeline(X, X_test)

cat_cols = ['Marital status', 'Application mode', 'Course', 'Nacionality', "Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Previous qualification"]
num_cols = [c for c in X_eng.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

print("Preprocessing...")
X_train_prep = preprocessor.fit_transform(X_eng)
X_test_prep = preprocessor.transform(X_test_eng)

# ==========================================
# 4. MANUAL RFE EXECUTION
# ==========================================
print("Running Manual RFE...")

# RFE Estimator: Balanced (agar fitur Enrolled tidak terbuang)
rfe_estimator = ManualLogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
rfe = ManualRFE(estimator=rfe_estimator, n_features_to_select=50, step=0.05)

rfe.fit(X_train_prep, y_encoded)
X_train_sel = rfe.transform(X_train_prep)
X_test_sel = rfe.transform(X_test_prep)

print(f"Features selected: {X_train_sel.shape[1]}")

# ==========================================
# 5. OPTIMAL WEIGHT SEARCH (THE LOGIC UPGRADE)
# ==========================================
print("\nOptimasi Threshold / Weights (Manual CV)...")

# Kita akan melakukan Manual Cross-Validation untuk mendapatkan probabilitas "Out-of-Fold".
# Ini lebih aman daripada cross_val_predict sklearn ketika menggunakan Custom Class manual.
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_probs = np.zeros((len(X_train_sel), 3)) # 3 Class: Dropout, Enrolled, Graduate

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_sel, y_encoded)):
    # Split data
    X_t, X_v = X_train_sel[train_idx], X_train_sel[val_idx]
    y_t, y_v = y_encoded[train_idx], y_encoded[val_idx]
    
    # Train Manual Model di fold ini
    # Note: Kita TIDAK pakai class_weight='balanced' di sini, karena kita akan menyeimbangkannya
    # lewat optimasi threshold nanti.
    model_cv = ManualLogisticRegression(C=0.8, max_iter=2000, tol=1e-4, random_state=42)
    model_cv.fit(X_t, y_t)
    
    # Predict Proba pada Validation set
    probs_v = model_cv.predict_proba(X_v)
    oof_probs[val_idx] = probs_v

# Fungsi Objektif untuk minimize (Mencari Macro F1 Terbaik)
def get_f1_score(weights):
    # weights adalah array [w_dropout, w_enrolled, w_graduate]
    final_probs = oof_probs * weights
    preds = np.argmax(final_probs, axis=1)
    return -f1_score(y_encoded, preds, average='macro')

# Cari bobot optimal menggunakan Nelder-Mead
print("Mencari bobot terbaik...")
init_weights = [1.0, 1.0, 1.0] # Start neutral
res = minimize(get_f1_score, init_weights, method='Nelder-Mead', tol=1e-6)
best_weights = res.x

print(f"Bobot Optimal Ditemukan: {best_weights}")

# Cek Skor F1 Macro Lokal dengan bobot baru
final_val_probs = oof_probs * best_weights
final_val_preds = np.argmax(final_val_probs, axis=1)
local_score = f1_score(y_encoded, final_val_preds, average='macro')

print("-" * 30)
print(f"ESTIMASI SKOR AKHIR (Local CV): {local_score:.5f}")
print("-" * 30)
print(classification_report(y_encoded, final_val_preds, target_names=classes))

# ==========================================
# 6. FINAL TRAINING & SUBMISSION
# ==========================================
print("\nTraining Final Manual Model (Full Data)...")

# Train model final pada SELURUH data latih
final_model = ManualLogisticRegression(
    C=0.8,
    max_iter=5000, 
    tol=1e-5,
    random_state=42
)

final_model.fit(X_train_sel, y_encoded)

# Prediksi Data Test
print("Predicting Test Data...")
test_probs = final_model.predict_proba(X_test_sel)

# APLIKASIKAN BOBOT OPTIMAL
test_probs_weighted = test_probs * best_weights
test_preds = np.argmax(test_probs_weighted, axis=1)
test_labels = le.inverse_transform(test_preds)

submission = pd.DataFrame({
    'Student_ID': test_ids,
    'Target': test_labels
})

filename = 'submission_manual_optimized.csv'
submission.to_csv(filename, index=False)
print(f"File '{filename}' created successfully.")