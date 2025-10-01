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

import warnings
warnings.filterwarnings('ignore')


#Set Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DATA_PATH = "./data/Train.csv" 


df = pd.read_csv(DATA_PATH) 

df.drop('user_id', axis=1, inplace=True) 
df.drop('MRG', axis=1, inplace=True)

mapping_dict = {'K > 24 month': 24, 'I 18-21 month': 18, 'H 15-18 month': 15, 'G 12-15 month': 12, 'J 21-24 month': 21, 'F 9-12 month': 9, 'E 6-9 month': 6, 'D 3-6 month': 3}
df['TENURE'] = df["TENURE"].apply(lambda x: mapping_dict[x])


original_columns = df.columns.tolist()
print("Original Columns:", original_columns)
# Numerical Features with highest feature importance (XgBoost)
num_features_high = ['REGULARITY', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE_RECH', 'MONTANT', 'ON_NET']
num_features_all = df.select_dtypes(include=["number"]).columns.tolist()
num_features_all.remove("CHURN")



def create_derived_features(df):
  #null counts
  df["NULL_COUNTS"] = df.isna().sum(axis=1)

  #revenue per frequency
  df['REVENUE_PER_FREQUENCE'] = df['REVENUE'] / (df['FREQUENCE'] + 1)

  #spending consistency
  df['SPENDING_CONSISTENCY'] = df['MONTANT'] / (df['REVENUE'] + 1)

  #high value customer
  df['HIGH_VALUE_CUSTOMER'] = (df['ARPU_SEGMENT'] > df['ARPU_SEGMENT'].median()).astype(int)

  #spending pattern
  df['SPENDING_PATTERN'] = df['MONTANT'] / (df['FREQUENCE_RECH'] + 1)

  # Engagement score
  df['ENGAGEMENT_SCORE'] = (df['FREQUENCE_RECH'] + df['FREQUENCE']) / 2 

  #usage decline
  df['USAGE_DECLINE'] = (df['FREQUENCE_RECH'] < df['FREQUENCE']).astype(int)


  # unlimited pack user
  df['UNLIMITED_PACK'] = 0
  df.loc[df['TOP_PACK'].str.contains("unlimited", case=False, na=False), 'UNLIMITED_PACK'] = 1

  # popular pack
  low = df['TOP_PACK'].value_counts().describe()["25%"]
  mid = df['TOP_PACK'].value_counts().describe()["50%"]
  high = df['TOP_PACK'].value_counts().describe()["75%"]

  val_counts = df['TOP_PACK'].value_counts()

  def pack_pop(pack):
    if pack is np.nan:
      return -1

    val = val_counts[pack]

    if val < low:
      return 1
    elif val < mid:
      return 2
    elif val < high:
      return 3
    else:
      return 4

  df["PACK_POP"] = df['TOP_PACK'].apply(pack_pop)


  manual_derived_columns = ['NULL_COUNTS', 'REVENUE_PER_FREQUENCE', 'SPENDING_CONSISTENCY',
                            'HIGH_VALUE_CUSTOMER', 'SPENDING_PATTERN', 'ENGAGEMENT_SCORE',
                            'USAGE_DECLINE', 'UNLIMITED_PACK', 'PACK_POP']
  return df, manual_derived_columns


df, manual_derived_columns = create_derived_features(df)


def brute_force_features(df, feature_list_all, feature_list_high):
  data = df.copy()
  n = len(feature_list_high)
  epsilon = 1e-6

  # Grouping by Region
  region_groups = data[feature_list_all + ['REGION']].groupby("REGION").mean()
  for feature in feature_list_all:
    feature_name = "REGION_MEAN_" + feature
    region_groups[feature_name] = region_groups[feature]
    region_groups.drop(feature, axis=1, inplace=True)

  data = pd.merge(data, region_groups, on='REGION', how='left')


  # log features
  for feature in feature_list_all:
    feature_name = 'log(' + feature + ")"
    data[feature_name] = np.log(data[feature] + epsilon)

  # Cross Features
  for i in range(n-1):
    for j in range(i+1, n):
      feature_add = feature_list_high[i] + "+" + feature_list_high[j]
      feature_sub = feature_list_high[i] + "-" + feature_list_high[j]
      feature_mult = feature_list_high[i] + "*" + feature_list_high[j]
      feature_div = feature_list_high[i] + "/" + feature_list_high[j]

      data[feature_add] = data[feature_list_high[i]] + data[feature_list_high[j]]
      data[feature_sub] = data[feature_list_high[i]] - data[feature_list_high[j]]
      data[feature_mult] = data[feature_list_high[i]] * data[feature_list_high[j]]
      data[feature_div] = data[feature_list_high[i]] / (data[feature_list_high[j]] + epsilon)

  return data


data = brute_force_features(df, num_features_all, num_features_high)

drop = ['REGION', 'TOP_PACK']
data = data.drop(drop, axis=1)


data.head()

X = data.drop('CHURN', axis=1)
y = data['CHURN']

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

PIPELINE_CACHE_DIR = 'tmp_cache'




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
'''

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

results = []
#results.append(evaluate_model(lr_grid, "Logistic Regression", X_val, y_val))
#results.append(evaluate_model(rf_grid, "Random Forest", X_val, y_val))
results.append(evaluate_model(xgb_grid, "XGB", X_val, y_val))

# Compare results
print(f"\n{'='*60}")
print(" MODEL COMPARISON SUMMARY ")
print(f"{'='*60}")

for result in results:
    print(f"{result['model']:25} | CV AUC: {result['best_cv_score']:.4f} | Test AUC: {result['test_auc']:.4f}")

