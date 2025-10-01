import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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


data.head()

X = data.drop('CHURN', axis=1)
y = data['CHURN']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)


# Define preprocessing for different data types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()


print("LR-- STARTED")

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])),
    ('feature_selector', SelectKBest(score_func=f_classif, k=20)),
    ('classifier', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
])

# Parameter grid for Logistic Regression:cite[8]
lr_param_grid = {
    'feature_selector__k': [25, 30, 40, 50, 60],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga']
}

# GridSearchCV for Logistic Regression:cite[1]
lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=StratifiedKFold(5), 
                      scoring='roc_auc', n_jobs=4, verbose=1)
lr_grid.fit(X_train, y_train)



print("RF-- STARTED")


# Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])),
    ('feature_selector', RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=20)),
    ('classifier', RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced'))
])

# Parameter grid for Random Forest:cite[3]:cite[6]
rf_param_grid = {
    'feature_selector__n_features_to_select': [25, 30, 40, 50, 60],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=StratifiedKFold(5), 
                      scoring='roc_auc', n_jobs=4, verbose=1)
rf_grid.fit(X_train, y_train)



print("KNN-- STARTED")

knn_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])),
    ('feature_selector', SelectKBest(score_func=f_classif, k=15)),
    ('classifier', KNeighborsClassifier())
])

# Parameter grid for KNN
knn_param_grid = {
    'feature_selector__k': [25, 30, 40, 50, 60],
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=StratifiedKFold(5), 
                       scoring='roc_auc', n_jobs=4, verbose=1)
knn_grid.fit(X_train, y_train)



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
    
    return {
        'model': model_name,
        'best_cv_score': grid_search.best_score_,
        'test_auc': test_auc,
        'best_params': grid_search.best_params_
    }

# Evaluate all models
results = []
results.append(evaluate_model(lr_grid, "Logistic Regression", X_val, y_val))
results.append(evaluate_model(rf_grid, "Random Forest", X_val, y_val))
results.append(evaluate_model(knn_grid, "K-Nearest Neighbors", X_val, y_val))

# Compare results
print(f"\n{'='*60}")
print(" MODEL COMPARISON SUMMARY ")
print(f"{'='*60}")

for result in results:
    print(f"{result['model']:25} | CV AUC: {result['best_cv_score']:.4f} | Test AUC: {result['test_auc']:.4f}")

