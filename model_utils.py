from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


from const import N_JOBS, DEVICE, PIPELINE_CACHE_DIR, RANDOM_SEED



#shared preprocessor
def get_shared_preprocessor(features):
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))  # keep sparse
        ]), features),
    ])


def get_models(features):
    shared_preprocessor = get_shared_preprocessor(features)
    return  {
            "RF" : {
                "pipeline" : Pipeline([
                            ('preprocessor', shared_preprocessor),
                            ('classifier', RandomForestClassifier(
                                random_state=RANDOM_SEED, class_weight='balanced', n_jobs=N_JOBS, verbose=1))
                            ], 
                            memory=PIPELINE_CACHE_DIR
                        ),
                "param_grid" : {
                            'classifier__n_estimators': [50, 100],
                            'classifier__max_depth': [10, 20, None],
                            'classifier__min_samples_split': [2, 5],
                            'classifier__min_samples_leaf': [1, 2]
                        }
            },
            "LR" : {
                "pipeline" : Pipeline([
                            ('preprocessor', shared_preprocessor),
                            ('feature_selector', SelectKBest(score_func=f_classif, k=20)),
                            ('classifier', LogisticRegression(random_state=RANDOM_SEED, n_jobs=N_JOBS, class_weight='balanced', max_iter=1000))
                            ],
                            memory=PIPELINE_CACHE_DIR
                        ),
                "param_grid" : {
                            'feature_selector__k': [25, 30, 40, 50, 60],
                            'classifier__C': [0.1, 1, 10],
                            'classifier__penalty': ['l1', 'l2'],
                            'classifier__max_iter': [5000],
                            'classifier__solver': ['liblinear', 'saga']
                        }
            },
            "XGB" : {
                "pipeline" : Pipeline([
                            ('preprocessor', shared_preprocessor),
                            ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=30)),
                            ('classifier', XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss', device=DEVICE))]
                            ),
                "param_grid" :  {
                            'feature_selector__k': [25, 30, 40],
                            'classifier__n_estimators': [100, 200],
                            'classifier__max_depth': [3, 5, 7],
                            'classifier__learning_rate': [0.01, 0.1, 0.2],
                            'classifier__subsample': [0.8, 1.0]
                        }
            }
        }