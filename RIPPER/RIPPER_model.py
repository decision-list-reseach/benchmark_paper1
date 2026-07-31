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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42 + i)

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

results_dir = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
ripper_rules_path = os.path.join(results_dir, 'ripper_rules.txt')

import re
ripper_model_step = pipeline.named_steps['ripper']
rule_str = str(ripper_model_step.ruleset_)
for idx, col_name in reversed(list(enumerate(X.columns))):
    rule_str = re.sub(rf'\b{idx}=', f'{col_name}=', rule_str)

# Reformat for readability
if rule_str.startswith('[['):
    rule_str = rule_str[2:]
if rule_str.endswith(']]'):
    rule_str = rule_str[:-2]
    
rules = rule_str.split('] V [')
formatted_rules = []

for i, r in enumerate(rules, 1):
    cond_str = r.replace('^', '\nAND ')
    # Clean up spaces around AND just in case
    cond_str = cond_str.replace(' \nAND  ', '\nAND ')
    cond_str = cond_str.replace(' \nAND ', '\nAND ')
    cond_str = cond_str.replace('\nAND  ', '\nAND ')
    
    # Format operators
    cond_str = cond_str.replace('=>', ' >= ').replace('=<', ' <= ')
    cond_str = re.sub(r'=(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)', r' ∈ [\1, \2]', cond_str)
    
    block = f'Rule {i}\nIF\n{cond_str}\n\nTHEN Churn\n\n------------------------\n'
    formatted_rules.append(block)

# Robust counting from internal representation
total_rules = len(ripper_model_step.ruleset_.rules)
total_conds = sum(len(rule.conds) for rule in ripper_model_step.ruleset_.rules)
avg_conds = total_conds / total_rules if total_rules > 0 else 0

header_block = f"""RIPPER Rule Set
================

Model parameters:
k = {ripper_model_step.k}
prune_size = {ripper_model_step.prune_size}
dl_allowance = {ripper_model_step.dl_allowance}
random_state = {ripper_model_step.random_state}

================================
Rules
================================

"""

stats_block = f"\n================================\n\nTotal rules: {total_rules}\n\nAverage conditions per rule: {avg_conds:.2f}\n\nTotal logical conditions: {total_conds}\n"

final_output = header_block + '\n'.join(formatted_rules) + stats_block

with open(ripper_rules_path, 'w') as f:
    f.write(final_output)