from const import BEST_RF_PARAMS, BEST_XGB_PARAMS, RESULTS_PATH, RANDOM_SEED, DATA_PATH, TEST_DATA_PATH, TARGET_COLUMN, BEST_LR_PARAMS
from model_utils import get_models
from features import feature_engineering

from sklearn.base import clone
import pandas as pd
import numpy as np
import random
import warnings

warnings.filterwarnings('ignore')


#Set Random seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


#load data
df = pd.read_csv(DATA_PATH) 
test_df = pd.read_csv(TEST_DATA_PATH)
test_df_user_ids = test_df['user_id']

#feature engineering
df = feature_engineering(df)
test_df = feature_engineering(test_df, split="test")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]
X_test = test_df

features_list = X.columns.tolist()

print("Data Preprocessing Completed - Training Started")
#---------------------------------------------------------
model_dict = get_models(features_list)

print("Random Forests training started with best Params")
#Random forests
rf_final = clone(model_dict["RF"]["pipeline"])
rf_final.set_params(**BEST_RF_PARAMS)   # params already use pipeline keys
rf_final.fit(X, y)

print("XGBoost training started with best Params")
xgb_final = clone(model_dict["RF"]["pipeline"])
xgb_final.set_params(**BEST_XGB_PARAMS)   # params already use pipeline keys
xgb_final.fit(X, y)


print("Logistic Regression training started with best Params")
lr_final = clone(model_dict["LR"]["pipeline"])
lr_final.set_params(**BEST_LR_PARAMS)   # params already use pipeline keys
lr_final.fit(X, y)



def predict_churn(model, model_name, X_test):
    """Do predictions and save"""
    print(f"\n{'='*50}")
    print(f" {model_name} ")
    print(f"{'='*50}")
    
    # Test set performance
    y_pred = model.predict(X_test)
        
    return {
        'model': model_name,
        'y_pred' : y_pred
    }


results = []

results.append(predict_churn(lr_final, "LR", X_test))
results.append(predict_churn(xgb_final, "XGBoost", X_test))
results.append(predict_churn(rf_final, "Random forest", X_test))

# Compare results
print(f"\n{'='*60}")
print(" MODEL COMPARISON SUMMARY ")
print(f"{'='*60}")

for result in results:
    if result['y_pred'] is not None:
        submission = pd.DataFrame({
            'user_id': test_df_user_ids,
            'CHURN': result['y_pred']
        })
        submission.to_csv(RESULTS_PATH + f"{result['model']}_result.csv", index=False)