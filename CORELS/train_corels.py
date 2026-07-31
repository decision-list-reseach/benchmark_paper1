import os
import sys
import time
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from corels import CorelsClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils    

def main():
    csv_path = os.path.join(parent_dir, 'data', 'data_ecommerce_customer_churn_corels.csv')
    
    print(f"Loading binarized data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    metrics_list = []
    
    for i in range(5):
        print(f"\n--- CORELS Run {i+1}/5 ---")
        # Stratified split to match XGBoost/RIPPER methodology without fixed random state for 5-fold evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42 + i)
        
        corels_clf = CorelsClassifier(c=0.01, n_iter=10000, verbosity=[], max_card=2)
        
        # Ensure dense numpy array
        X_train_dense = np.asarray(X_train.values, dtype=np.uint8)
        y_train_dense = np.asarray(y_train.values, dtype=np.uint8)
        X_test_dense = np.asarray(X_test.values, dtype=np.uint8)
        
        start = time.perf_counter()
        corels_clf.fit(X_train_dense, y_train_dense, features=X.columns.tolist(), prediction_name="Churn")
        end = time.perf_counter()
        time_needed = end - start
        
        preds = corels_clf.predict(X_test_dense)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        metrics_list.append({
            "accuracy": acc,
            "churn_precision": prec,
            "churn_recall": rec,
            "f1": f1,
            "time_needed": time_needed
        })
        
        # Save scores using the utils function
        utils.save_scores("scores_from_corels", current_dir, y_test, preds, time_needed)

    print("\nAveraged metrics from 5 results:")
    df_metrics = pd.DataFrame(metrics_list)
    print(df_metrics.mean().to_string())
    
    rule_list_str = corels_clf.rl()
    print("\n### Provably Optimal Rule List (Last Run) ###")
    print(rule_list_str)
    
    # Save Rule List
    results_dir = os.path.join(parent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    rules_path = os.path.join(results_dir, 'corels_rules.txt')
    
    # Calculate stats
    lines = str(rule_list_str).strip().split('\n')
    total_rules = 0
    total_conds = 0
    for line in lines:
        if line.startswith('if ') or line.startswith('else if '):
            total_rules += 1
            conds_in_rule = line.count('&&') + 1
            total_conds += conds_in_rule
            
    avg_conds = total_conds / total_rules if total_rules > 0 else 0
    
    header_block = f"""CORELS Rule List
================

Parameters:
c = {corels_clf.c}
max_card = {corels_clf.max_card}
policy = {corels_clf.policy}
n_iter = {corels_clf.n_iter}

================================
Rule List
================================

"""

    stats_block = f"\n================================\n\nTotal rules: {total_rules}\n\nAverage conditions per rule: {avg_conds:.2f}\n\nTotal logical conditions: {total_conds}\n"
    final_output = header_block + str(rule_list_str) + stats_block

    with open(rules_path, "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    main()
