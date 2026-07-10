import pandas as pd
import wittgenstein as lw
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import save_scores

csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn.csv')
df = pd.read_csv(csv_path)

df = pd.get_dummies(df, columns=["PreferedOrderCat", "MaritalStatus"])

y = df['Churn']
X = df.drop('Churn', axis=1)

imputer = SimpleImputer(strategy='median')
smote = SMOTE(random_state=42)
ripper_model = lw.RIPPER(
    k=1,
    prune_size=0.33,
    dl_allowance=128,
    random_state=42
 )
#Best RIPPER Parameters: {'ripper__dl_allowance': 128, 'ripper__k': 1, 'ripper__prune_size': 0.33}

pipeline = ImbPipeline(steps=[
    ('imputer', imputer),
    ('smote', smote),
    ('ripper', ripper_model)
])


for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

    print("Training RIPPER...")
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    end = time.perf_counter()

    
    print("\n--- RIPPER Decision Rules ---")
    pipeline.named_steps['ripper'].out_model()


    preds = pipeline.predict(X_test)
    time_needed = end-start

    train_pred = pipeline.predict(X_train)
    print(f1_score(y_train, train_pred))

    save_scores("scores_from_ripper", current_dir, y_test, preds, time_needed)