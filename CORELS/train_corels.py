import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from corels import CorelsClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils    

import json
import numpy as np

def main():
    csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn_corels.csv')
    
    print(f"Loading binarized data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # Stratified split to match XGBoost methodology
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Training CORELS Classifier...")
    # c is the regularization parameter. Smaller c -> shorter rules.
    corels_clf = CorelsClassifier(c=0.01, n_iter=10000, verbosity=[], max_card=2)
    
    start = time.perf_counter()
    # Ensure X_train is a dense numpy array
    # The preprocessing pipeline guarantees dense features via onehot-dense and sparse_output=False
    X_train_dense = np.asarray(X_train.values, dtype=np.uint8)
    y_train_dense = np.asarray(y_train.values, dtype=np.uint8)
    
    corels_clf.fit(X_train_dense, y_train_dense, features=X.columns.tolist(), prediction_name="Churn")
    end = time.perf_counter()
    time_needed = end - start
    
    print(f"Training completed in {time_needed:.2f} seconds.\n")
    
    rule_list_str = corels_clf.rl()
    print("### Provably Optimal Rule List ###")
    print(rule_list_str)
    print("##################################\n")
    
    rules_obj = rule_list_str.rules
    num_rules = len(rules_obj)
    if num_rules > 1:
        avg_rule_length = sum(len(r["antecedents"]) for r in rules_obj[:-1]) / (num_rules - 1)
    else:
        avg_rule_length = 0
        
    print(f"Number of Rules (including default): {num_rules}")
    print(f"Average Rule Length (conditions per IF statement): {avg_rule_length:.2f}\n")
    
    # Prepare results directory
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save Rule List
    rules_path = os.path.join(results_dir, 'corels_rules.txt')
    with open(rules_path, "w") as f:
        f.write(str(rule_list_str))
        
    X_test_dense = np.asarray(X_test.values, dtype=np.uint8)
    preds = corels_clf.predict(X_test_dense)
    train_pred = corels_clf.predict(X_train_dense)
    
    # Compute Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    
    print("Confusion Matrix:")
    print(cm)
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Training_Time_Seconds": time_needed,
        "Number_of_Rules": num_rules,
        "Average_Rule_Length": avg_rule_length,
        "Confusion_Matrix": cm.tolist()
    }
    
    metrics_path = os.path.join(results_dir, 'corels_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, preds))
    print(f"\nRules saved to: {rules_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Save scores using the utils function if desired
    try:
        utils.save_scores("scores_from_corels", current_dir, y_test, preds, time_needed)
        print("Scores saved successfully.")
    except Exception as e:
        print(f"Could not save scores via utils: {e}")

if __name__ == "__main__":
    main()
