import pandas as pd
import numpy as np
import os
from .normalize import normalize_name, normalize_address
from .candidates import generate_tfidf_candidates
from .features import generate_features

def create_predictions(df_s1, df_s2, df_s3, models, threshold=0.5):
    # Normalize
    for df in [df_s1, df_s2, df_s3]:
        df['norm_name'] = df['business_name'].apply(normalize_name)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
        
    # Generate candidates S1 -> S2
    print("Generating candidates for S1 -> S2...")
    cand_s2 = generate_tfidf_candidates(df_s1, df_s2, 'norm_name', 'S2', k=20)
    
    # Generate candidates S1 -> S3
    print("Generating candidates for S1 -> S3...")
    cand_s3 = generate_tfidf_candidates(df_s1, df_s3, 'norm_name', 'S3', k=20)
    
    # Save candidate pairs (optional, for debugging/rules)
    # pd.concat([cand_s2, cand_s3]).to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
    
    # Feature engineering
    print("Generating features S1 -> S2...")
    feat_s2 = generate_features(cand_s2, df_s1, df_s2, 'S2')
    feat_s2['target_entity_id'] = feat_s2['S2_entity_id']
    
    print("Generating features S1 -> S3...")
    feat_s3 = generate_features(cand_s3, df_s1, df_s3, 'S3')
    feat_s3['target_entity_id'] = feat_s3['S3_entity_id']
    
    # Combine features
    all_feats = pd.concat([
        feat_s2.drop(columns=['S2_entity_id']), 
        feat_s3.drop(columns=['S3_entity_id'])
    ])
    
    X = all_feats.drop(columns=['source1_entity_id', 'target_entity_id'])
    X['country_1'] = X['country_1'].astype(str).fillna('-1')
    
    # Predict with ensemble
    print("Predicting...")
    preds = np.zeros(len(X))
    for model in models:
        preds += model.predict_proba(X)[:, 1]
    preds /= len(models)
    
    all_feats['prob'] = preds
    
    # Thresholding
    matches = all_feats[all_feats['prob'] >= threshold]
    
    # Group by source1_entity_id
    results = matches.groupby('source1_entity_id')['target_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    results.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure all S1 IDs are present
    final_results = df_s1[['entity_id']].merge(results, left_on='entity_id', right_on='source1_entity_id', how='left')
    final_results = final_results.fillna('')
    final_results = final_results[['entity_id', 'matched_entity_ids']]
    final_results.columns = ['source1_entity_id', 'matched_entity_ids']
    
    cand_s2['target_entity_id'] = cand_s2['S2_entity_id']
    cand_s3['target_entity_id'] = cand_s3['S3_entity_id']
    candidate_pairs = pd.concat([
        cand_s2[['source1_entity_id', 'target_entity_id']], 
        cand_s3[['source1_entity_id', 'target_entity_id']]
    ])
    return final_results, candidate_pairs

if __name__ == '__main__':
    pass
