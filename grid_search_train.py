from features import feature_engineering
from const import RANDOM_SEED, DATA_PATH, TARGET_COLUMN
from model_utils import get_models, N_JOBS

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import HalvingGridSearchCV
import warnings



warnings.filterwarnings('ignore')


#Set Random seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

df = pd.read_csv(DATA_PATH) 

#feature engineering
df = feature_engineering(df)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
features_list = X.columns.tolist()

print("Data Preprocessing Completed - Training Started")
#---------------------------------------------------------

model = get_models(features_list)

print("Random Forest GRID SEARCH -- STARTED")
rf_grid = HalvingGridSearchCV(
    model["RF"]["pipeline"], 
    model["RF"]["param_grid"], cv=StratifiedKFold(3), scoring='roc_auc',
    n_jobs=N_JOBS, verbose=2, factor=3, refit=True
)
rf_grid.fit(X_train, y_train)


print("XGBOOST GRID SEARCH -- STARTED")
xgb_grid = HalvingGridSearchCV(
    model["XGB"]["pipeline"], 
    model["XGB"]["param_grid"], cv=StratifiedKFold(3), scoring='roc_auc',
    n_jobs=N_JOBS, verbose=2, factor=3, refit=True
)
xgb_grid.fit(X_train, y_train)


print("LR GRID SEARCH -- STARTED")
cv = StratifiedKFold(3)
lr_grid = HalvingGridSearchCV(
    estimator=model["LR"]["PIPELINE"],
    param_grid=model["LR"]["param_grid"],
    cv=cv,
    scoring='roc_auc',
    n_jobs=N_JOBS,
    verbose=1,
    resource='n_samples',
    factor=3,
    min_resources='exhaust',
    aggressive_elimination=False,
    refit=True
)
lr_grid.fit(X_train, y_train)




# ---------- Evaluation----------
def evaluate_model(grid_search, model_name, X_test, y_test):
    """Evaluate and display model performance"""
    print(f"\n{'='*50}")
    print(f" {model_name} ")
    print(f"{'='*50}")
    
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Test set performance
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test AUC: {test_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

    print("Confusion matrix:")
    print(f"CM : {cm}")
    print(f"CM_NORM : {cm_norm}")
    
    return {
        'model': model_name,
        'best_cv_score': grid_search.best_score_,
        'test_auc': test_auc,
        'best_params': grid_search.best_params_
    }


results = []
results.append(evaluate_model(lr_grid, "Logistic Regression", X_val, y_val))
results.append(evaluate_model(rf_grid, "Random Forest", X_val, y_val))
results.append(evaluate_model(xgb_grid, "XGBoost", X_val, y_val))

# Compare results
print(f"\n{'='*60}")
print(" MODEL COMPARISON SUMMARY ")
print(f"{'='*60}")


for result in results:
    print(f"{result['model']:25} | CV AUC: {result['best_cv_score']:.4f} | Test AUC: {result['test_auc']:.4f}")





