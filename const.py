RANDOM_SEED = 42
DATA_PATH = "./data/Train.csv"
TEST_DATA_PATH = "./data/Test.csv"
TARGET_COLUMN = "CHURN"
PIPELINE_CACHE_DIR = 'tmp_cache'

LOG_FILE_PATH = "./logs/gridsearch.out"

BEST_RF_PARAMS = {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}

BEST_XGB_PARAMS = {'classifier__learning_rate': 0.1, 'classifier__max_depth': 3, 'classifier__n_estimators': 200, 'classifier__subsample': 1.0, 'feature_selector__k': 30}

RESULTS_PATH = "./results/predictions{}.csv"
