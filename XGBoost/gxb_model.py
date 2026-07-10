import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils

csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn.csv')

df = pd.read_csv(csv_path)
df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])

y = df['Churn']
X = df.drop('Churn', axis=1)

xgb = XGBClassifier(
    # Used GridSearchCv to find optimal parameters for F1 score
    n_estimators=200,
    max_depth=4,
    learning_rate=0.2,
    scale_pos_weight=3,
    subsample=0.8,
    colsample_bytree=1,
    random_state=42
    )


for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

    start = time.perf_counter() #Meassure time needed to train XGBoost
    xgb.fit(X_train, y_train)
    end = time.perf_counter()

    preds = xgb.predict(X_test)
    time_needed = end-start

    train_pred = xgb.predict(X_train)
    print(f1_score(y_train, train_pred))
    
    utils.save_scores("scores_from_xgb1", current_dir, y_test, preds, time_needed)