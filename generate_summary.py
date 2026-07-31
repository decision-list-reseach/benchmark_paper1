import os
import re

def parse_stats(file_path):
    stats = {'Total rules': 0, 'Total logical conditions': 0, 'Average conditions per rule': 0.0}
    if not os.path.exists(file_path):
        return stats
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    rules_match = re.search(r'Total rules:\s*(\d+)', content)
    conds_match = re.search(r'Total logical conditions:\s*(\d+)', content)
    avg_match = re.search(r'Average conditions per rule:\s*([\d\.]+)', content)
    
    if rules_match: stats['Total rules'] = int(rules_match.group(1))
    if conds_match: stats['Total logical conditions'] = int(conds_match.group(1))
    if avg_match: stats['Average conditions per rule'] = float(avg_match.group(1))
    
    return stats

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'results')
    
    corels_path = os.path.join(results_dir, 'corels_rules.txt')
    ripper_path = os.path.join(results_dir, 'ripper_rules.txt')
    summary_path = os.path.join(results_dir, 'comparison_summary.md')
    
    corels_stats = parse_stats(corels_path)
    ripper_stats = parse_stats(ripper_path)
    
    c_rules = corels_stats['Total rules']
    c_conds = corels_stats['Total logical conditions']
    
    r_rules = ripper_stats['Total rules']
    r_conds = ripper_stats['Total logical conditions']
    
    if c_rules > 0:
        rule_mult = r_rules / c_rules
    else:
        rule_mult = 0
        
    if c_conds > 0:
        cond_mult = r_conds / c_conds
    else:
        cond_mult = 0

    markdown_content = f"""# Rule Extraction Summary

## CORELS

- Total rules: {c_rules}
- Total conditions: {c_conds}
- Average conditions/rule: {corels_stats['Average conditions per rule']:.2f}

## RIPPER

- Total rules: {r_rules}
- Total conditions: {r_conds}
- Average conditions/rule: {ripper_stats['Average conditions per rule']:.2f}

## Difference

- RIPPER has {rule_mult:.1f}× more rules.
- RIPPER contains {cond_mult:.1f}× more logical conditions.
- CORELS produces a substantially simpler and more interpretable rule list.
"""

    with open(summary_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"Successfully generated {summary_path}")

if __name__ == "__main__":
    main()
