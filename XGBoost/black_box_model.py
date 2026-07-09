import pandas as pd
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'data', 'data_ecommerce_customer_churn.csv')
df = pd.read_csv(csv_path)
# print(df.head())
df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])
# df.info()

y = df['Churn']
X = df.drop('Churn', axis=1)

"""
    Optimized parameters for Precision, for Accuracy we got the same paratemers 
    n_estimators=200,
    max_depth=4,
    learning_rate=0.2,
    scale_pos_weight=3,
    random_state=42
"""
"""
    Optimized parameters with GridSearchCV to get the best result on Recall
    n_estimators=50,
    max_depth=2,
    learning_rate=0.2,
    scale_pos_weight=5,
    random_state=42
"""

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.2,
    scale_pos_weight=3,
    random_state=42
    )


for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    start = time.perf_counter() #Meassure time needed to train XGBoost
    xgb.fit(X_train, y_train)
    end = time.perf_counter()

    preds = xgb.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc * 100:.2f}%\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\n")

    print("Classification Report:")
    print(classification_report(y_test, preds))


    params_and_metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "churn_precision": precision_score(y_test, preds, pos_label=1),
        "churn_recall": recall_score(y_test, preds, pos_label=1),
        "time_needed": (end-start)
    }

    df_current_run = pd.DataFrame([params_and_metrics])
    log_file = os.path.join(current_dir, 'logs', 'file_name.csv')

    if not os.path.isfile(log_file):
        df_current_run.to_csv(log_file, index=False)
    else:
        df_current_run.to_csv(log_file, mode='a', header=False, index=False)