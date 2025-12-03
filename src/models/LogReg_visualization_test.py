
# Minimal test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.datasets import load_iris
from LogRegression import LogRegression

X, y = load_iris(return_X_y=True)
model = LogRegression(solver='batch', max_iter=50)
model.fit(X, y)
model.animate_logloss(X, y, param_indices=(0, 4))