import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.base import clone


from features import feature_engineering
from const import RANDOM_SEED, DATA_PATH, TARGET_COLUMN, TEST_DATA_PATH, PIPELINE_CACHE_DIR
from const import BEST_RF_PARAMS, BEST_XGB_PARAMS, RESULTS_PATH

import warnings
warnings.filterwarnings('ignore')


print("ML Project Started" )


#Set Random seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


df = pd.read_csv(DATA_PATH) 
test_df = pd.read_csv(TEST_DATA_PATH)

#feature engineering
df = feature_engineering(df)
test_df = feature_engineering(test_df, split="test")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_test = test_df


#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)


numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

#shared preprocessor
shared_preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))  # keep sparse
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01))
    ]), categorical_features)
])


print("Data Preprocessing Completed - Training Started")


print("Random Forest -- STARTED")
rf_pipeline = Pipeline([
    ('preprocessor', shared_preprocessor),
    # ('feature_selector', SelectFromModel(RandomForestClassifier(
    #     random_state=RANDOM_SEED, n_estimators=100, n_jobs=-1, class_weight='balanced'
    # ))),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, verbose=1
    ))
], memory=PIPELINE_CACHE_DIR)


rf_final = clone(rf_pipeline)
rf_final.set_params(**BEST_RF_PARAMS)   # params already use pipeline keys
rf_final.fit(X_train, y_train)


print("XGBOOST-- STARTED")
# XGBoost Pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', shared_preprocessor),
    ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=30)),
    ('classifier', XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', device='cuda:0'))
])

xgb_final = clone(xgb_pipeline)
xgb_final.set_params(**BEST_XGB_PARAMS)   # params already use pipeline keys
xgb_final.fit(X_train, y_train)



'''
# RANDOM FOREST
print("RF -- STARTED")

rf_pipeline = Pipeline([
    ('preprocessor', shared_preprocessor),
    # ('feature_selector', SelectFromModel(RandomForestClassifier(
    #     random_state=RANDOM_SEED, n_estimators=100, n_jobs=-1, class_weight='balanced'
    # ))),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1, verbose=1
    ))
], memory=PIPELINE_CACHE_DIR)

rf_param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
rf_grid = HalvingGridSearchCV(
    rf_pipeline, rf_param_grid, cv=3, scoring='roc_auc',
    n_jobs=6, verbose=10, factor=3, refit=True
)
rf_grid.fit(X_train, y_train)


print("XGBOOST-- STARTED")

# XGBoost Pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', shared_preprocessor),
    ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=30)),
    ('classifier', XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', device='cuda:0'))
])

xgb_param_grid = {
     'feature_selector__k': [25, 30, 40],
     'classifier__n_estimators': [100, 200],
     'classifier__max_depth': [3, 5, 7],
     'classifier__learning_rate': [0.01, 0.1, 0.2],
     'classifier__subsample': [0.8, 1.0]
 }

# # GridSearchCV for XGBoost
xgb_grid = HalvingGridSearchCV(xgb_pipeline, xgb_param_grid, cv=StratifiedKFold(3),
                         scoring='roc_auc', n_jobs=6, verbose=2)

xgb_grid.fit(X_train, y_train)
'''
'''
# ---------- Evaluation (unchanged) ----------
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
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrices
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')

    print(f"Confusion matrix:")
    print(f"CM : {cm}")
    print(f"CM_NORM : {cm_norm}")
    
    return {
        'model': model_name,
        'best_cv_score': grid_search.best_score_,
        'test_auc': test_auc,
        'best_params': grid_search.best_params_
    }
'''


def evaluate_model(model, model_name, X_test, y_test = None):
    """Evaluate and display model performance"""
    print(f"\n{'='*50}")
    print(f" {model_name} ")
    print(f"{'='*50}")
    
    # Test set performance
    y_pred = model.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

    best_cv_score = None
    test_auc = None

    if y_test is not None:
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrices
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')

        print(f"Confusion matrix:")
        print(f"CM : {cm}")
        print(f"CM_NORM : {cm_norm}")
        y_pred = None
        
    return {
        'model': model_name,
        'best_cv_score': grid_search.best_score_,
        'test_auc': test_auc,
        'y_pred' : y_pred
    }

results = []
#results.append(evaluate_model(lr_grid, "Logistic Regression", X_val, y_val))
#results.append(evaluate_model(rf_grid, "Random Forest", X_val, y_val))
results.append(evaluate_model(rf_final, "Random forest", X_val, y_val))
results.append(evaluate_model(xgb_final, "XGBoost", X_val, y_val))
results.append(evaluate_model(xgb_final, "XGBoost", X_test))
results.append(evaluate_model(rf_final, "Random forest", X_test))

# Compare results
print(f"\n{'='*60}")
print(" MODEL COMPARISON SUMMARY ")
print(f"{'='*60}")

for result in results:
    print(f"{result['model']:25} | CV AUC: {result['best_cv_score']:.4f} | Test AUC: {result['test_auc']:.4f}")
    if y_pred is not None:
        submission = pd.DataFrame({
            'ID': X_test.index,
            'CHURN_PROBABILITY': result['y_pred']
        })
        submission.to_csv(RESULTS_PATH.format(result[model]), index=False)

