import pandas as pd

def calculate_f05(y_true, y_pred):
    """
    Calculate Macro F0.5 for entity resolution.
    y_true: dict of source1 -> set of matches
    y_pred: dict of source1 -> set of matches
    """
    f05_scores = []
    
    for s1, true_matches in y_true.items():
        pred_matches = y_pred.get(s1, set())
        
        # If true is empty
        if not true_matches:
            if not pred_matches:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        # If pred is empty but true is not
        if not pred_matches:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_matches.intersection(pred_matches))
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
        f05_scores.append(f05)
        
    return sum(f05_scores) / len(f05_scores) if f05_scores else 0.0

def evaluate_predictions(gt_path, pred_path):
    gt = pd.read_csv(gt_path, sep='\t').fillna('')
    pred = pd.read_csv(pred_path, sep='\t').fillna('')
    
    y_true = {}
    for _, row in gt.iterrows():
        matches = row['matched_entity_ids']
        y_true[row['source1_entity_id']] = set(matches.split(',')) if matches else set()
        
    y_pred = {}
    for _, row in pred.iterrows():
        matches = row['matched_entity_ids']
        y_pred[row['source1_entity_id']] = set(matches.split(',')) if matches else set()
        
    f05 = calculate_f05(y_true, y_pred)
    return f05

if __name__ == '__main__':
    # usage example
    pass
