import pandas as pd
import numpy as np
import random
from const import RANDOM_SEED, DATA_PATH

#Set Random seed
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



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



def brute_force_features(data, feature_list_all, feature_list_high):
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


def feature_engineering(df, split = "train"):
  df.drop('user_id', axis=1, inplace=True) 
  #invariant
  df.drop('MRG', axis=1, inplace=True)

  #High NULL values
  df.drop('ZONE1', axis=1, inplace=True) 
  df.drop('ZONE2', axis=1, inplace=True) 

  # Convert TENURE to numeric
  mapping_dict = {'K > 24 month': 24, 'I 18-21 month': 18, 'H 15-18 month': 15, 'G 12-15 month': 12, 'J 21-24 month': 21, 'F 9-12 month': 9, 'E 6-9 month': 6, 'D 3-6 month': 3}
  df['TENURE'].fillna('K > 24 month', inplace=True)
  df['TENURE'] = df["TENURE"].apply(lambda x: mapping_dict[x])

  original_columns = df.columns.tolist()
  #print("Original Columns:", original_columns)
  # Numerical Features with highest feature importance (XgBoost)
  num_features_high = ['REGULARITY', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE_RECH', 'MONTANT', 'ON_NET']
  num_features_all = df.select_dtypes(include=["number"]).columns.tolist()
  if(split=="train"):
    num_features_all.remove("CHURN")

  df, manual_derived_columns = create_derived_features(df)

  data = brute_force_features(df, num_features_all, num_features_high)

  drop = ['REGION', 'TOP_PACK']
  data = data.drop(drop, axis=1)

  return data
