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



# KNN
print("KNN -- STARTED")

knn_pipeline = Pipeline([
    ('preprocessor', shared_preprocessor),
    ('svd', TruncatedSVD(n_components=100, random_state=RANDOM_SEED)),
    ('classifier', KNeighborsClassifier())
], memory=PIPELINE_CACHE_DIR)

knn_param_grid = {
    'svd__n_components': [50, 100, 150],     
    'classifier__n_neighbors': [5, 11, 21],  
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

knn_grid = HalvingGridSearchCV(
    knn_pipeline, knn_param_grid, cv=cv3, scoring='roc_auc',
    n_jobs=-1, verbose=10, factor=3, refit=True
)
knn_grid.fit(X_train, y_train)



# LOGISTIC REGRESSION
print("LR -- STARTED")

lr_pipeline = Pipeline([
    ('preprocessor', shared_preprocessor),
    ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=25)),
    ('classifier', LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000, solver='saga', n_jobs=6, verbose=0
    ))
], memory=PIPELINE_CACHE_DIR)

# Leaner grid + successive halving to skip weak configs early
lr_param_grid = {
    'feature_selector__k': [25],
    'classifier__C': [0.05],
    'classifier__penalty': ['l1']
}

cv3 = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)

# Using HalvingGridSearchCV for efficiency
lr_grid = HalvingGridSearchCV(
    lr_pipeline, lr_param_grid, cv=cv3, scoring='roc_auc',
    n_jobs=6, verbose=3, factor=3, refit=True
)
lr_grid.fit(X_train, y_train)
