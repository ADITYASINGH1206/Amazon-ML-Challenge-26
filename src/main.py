import pandas as pd
import numpy as np
import os
import argparse
import joblib
import gc

from src.normalize import normalize_name, normalize_address
from src.candidates import generate_tfidf_candidates
from src.features import generate_features
from src.train import train_model
from src.predict import create_predictions

def prepare_training_data(train_dir, sample=None):
    print("Loading training data...")
    train_s1 = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', nrows=sample).fillna('')
    train_s2 = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', nrows=sample*3 if sample else None).fillna('')
    train_s3 = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', nrows=sample*3 if sample else None).fillna('')
    gt = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t').fillna('')
    
    if sample:
        s1_ids = set(train_s1['entity_id'])
        gt = gt[gt['source1_entity_id'].isin(s1_ids)]
        # For S2/S3 we might just sample randomly or keep all since TF-IDF sparse dot product needs a corpus
        # We will keep a small subset for S2/S3 if sampling to speed up
        train_s2 = train_s2.head(sample * 3)
        train_s3 = train_s3.head(sample * 3)
        
    for df in [train_s1, train_s2, train_s3]:
        df['norm_name'] = df['business_name'].apply(normalize_name)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
        
    print("Generating candidates for training S1 -> S2...")
    cand_s2 = generate_tfidf_candidates(train_s1, train_s2, 'norm_name', 'S2', k=5)
    print("Generating candidates for training S1 -> S3...")
    cand_s3 = generate_tfidf_candidates(train_s1, train_s3, 'norm_name', 'S3', k=5)
    
    cand_s2['target_entity_id'] = cand_s2['S2_entity_id']
    cand_s3['target_entity_id'] = cand_s3['S3_entity_id']
    
    candidates = pd.concat([
        cand_s2[['source1_entity_id', 'target_entity_id', 'score']],
        cand_s3[['source1_entity_id', 'target_entity_id', 'score']]
    ])
    
    # Expand ground truth
    gt_pairs = []
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if matches:
            for match in matches.split(','):
                gt_pairs.append({'source1_entity_id': s1, 'target_entity_id': match, 'label': 1})
                
    df_gt = pd.DataFrame(gt_pairs)
    
    # Merge candidates with ground truth to create labels
    if not df_gt.empty:
        df_train_pairs = candidates.merge(df_gt, on=['source1_entity_id', 'target_entity_id'], how='left')
        df_train_pairs['label'] = df_train_pairs['label'].fillna(0)
        
        # Add positives that were missed by candidate generation
        missed_gt = df_gt.merge(candidates, on=['source1_entity_id', 'target_entity_id'], how='left', indicator=True)
        missed_gt = missed_gt[missed_gt['_merge'] == 'left_only'].drop(columns=['_merge'])
        missed_gt['score'] = 1.0 # arbitrary score
        
        df_train_pairs = pd.concat([df_train_pairs, missed_gt], ignore_index=True)
    else:
        df_train_pairs = candidates.copy()
        df_train_pairs['label'] = 0
        
    print(f"Training pairs: {len(df_train_pairs)} (Positives: {df_train_pairs['label'].sum()})")
    
    # Generate features
    print("Generating training features...")
    # Split back into S2 and S3 for feature generation
    pairs_s2 = df_train_pairs[df_train_pairs['target_entity_id'].str.startswith('S2-')]
    pairs_s3 = df_train_pairs[df_train_pairs['target_entity_id'].str.startswith('S3-')]
    
    pairs_s2_renamed = pairs_s2.rename(columns={'target_entity_id': 'S2_entity_id'})
    pairs_s3_renamed = pairs_s3.rename(columns={'target_entity_id': 'S3_entity_id'})
    
    feat_s2 = generate_features(pairs_s2_renamed, train_s1, train_s2, 'S2')
    feat_s3 = generate_features(pairs_s3_renamed, train_s1, train_s3, 'S3')
    
    feat_s2['target_entity_id'] = feat_s2['S2_entity_id']
    feat_s3['target_entity_id'] = feat_s3['S3_entity_id']
    
    df_features = pd.concat([
        feat_s2.drop(columns=['S2_entity_id']), 
        feat_s3.drop(columns=['S3_entity_id'])
    ])
    
    df_features = df_features.merge(df_train_pairs[['source1_entity_id', 'target_entity_id', 'label']], on=['source1_entity_id', 'target_entity_id'], how='left')
    return df_features

def main(train_dir, test_dir, output_dir, model_dir, sample=None, mode='all'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    models = []
    
    if mode in ['train', 'all']:
        df_train_features = prepare_training_data(train_dir, sample=sample)
        print("Training models...")
        labels = df_train_features['label']
        groups = df_train_features['source1_entity_id']
        features = df_train_features.drop(columns=['label'])
        
        models = train_model(features, labels, groups, cat_features=['country_1'])
        
        # Save models
        for i, model in enumerate(models):
            joblib.dump(model, os.path.join(model_dir, f'model_fold_{i}.joblib'))
        print(f"Saved {len(models)} models to {model_dir}")
        
        # Clean up memory
        del df_train_features
        gc.collect()
        
    if mode in ['predict', 'all']:
        if not models:
            # Load models
            print("Loading models...")
            for i in range(5):
                model_path = os.path.join(model_dir, f'model_fold_{i}.joblib')
                if os.path.exists(model_path):
                    models.append(joblib.load(model_path))
                    
        if not models:
            raise ValueError("No models found to run prediction.")
            
        print("Loading test data...")
        test_s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', nrows=sample).fillna('')
        test_s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', nrows=sample*3 if sample else None).fillna('')
        test_s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', nrows=sample*3 if sample else None).fillna('')
        
        final_results, candidate_pairs = create_predictions(test_s1, test_s2, test_s3, models, threshold=0.5)
        
        candidates_agg = candidate_pairs.groupby('source1_entity_id')['target_entity_id'].apply(lambda x: ','.join(x.dropna().unique())).reset_index()
        candidates_agg.columns = ['source1_entity_id', 'candidate_entity_ids']
        
        # Ensure all test_s1 entities are included
        final_candidates = test_s1[['entity_id']].merge(candidates_agg, left_on='entity_id', right_on='source1_entity_id', how='left')
        final_candidates['candidate_entity_ids'] = final_candidates['candidate_entity_ids'].fillna('')
        final_candidates = final_candidates[['entity_id', 'candidate_entity_ids']]
        final_candidates.columns = ['source1_entity_id', 'candidate_entity_ids']
        
        candidate_pairs_path = os.path.join(output_dir, 'candidate_pairs.tsv')
        final_candidates.to_csv(candidate_pairs_path, sep='\t', index=False)
        print(f"Saved candidates to {candidate_pairs_path}")
        
        results_path = os.path.join(output_dir, 'matching_results.tsv')
        final_results.to_csv(results_path, sep='\t', index=False)
        print(f"Saved final matches to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='student_resource/dataset/train')
    parser.add_argument('--test-dir', default='student_resource/dataset/test')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--sample', type=int, default=None, help='Run on a sample for testing')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'predict', 'all'])
    args = parser.parse_args()
    
    main(args.train_dir, args.test_dir, args.output_dir, args.model_dir, args.sample, args.mode)
