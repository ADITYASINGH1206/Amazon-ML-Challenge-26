import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

def calculate_similarity_features(row, col1, col2):
    s1 = str(row[col1]) if pd.notnull(row[col1]) else ""
    s2 = str(row[col2]) if pd.notnull(row[col2]) else ""
    
    if not s1 or not s2:
        return [0.0, 0.0, 0.0, 0.0]
        
    return [
        fuzz.ratio(s1, s2) / 100.0,
        fuzz.token_set_ratio(s1, s2) / 100.0,
        fuzz.token_sort_ratio(s1, s2) / 100.0,
        distance.JaroWinkler.normalized_similarity(s1, s2)
    ]

def generate_features(df_pairs, df_s1, df_target, target_prefix):
    """
    df_pairs: candidate pairs with source1_entity_id and target_entity_id
    df_s1: Source 1 dataframe
    df_target: Target dataframe (S2 or S3)
    target_prefix: 'S2' or 'S3'
    """
    # Merge information
    df = df_pairs.merge(df_s1, left_on='source1_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name_1', 'business_address': 'address_1', 'country': 'country_1'})
    
    df = df.merge(df_target, left_on=f'{target_prefix}_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name_2', 'business_address': 'address_2', 'country': 'country_2'})
    
    # Calculate similarities
    name_sims = df.apply(lambda row: calculate_similarity_features(row, 'name_1', 'name_2'), axis=1, result_type='expand')
    name_sims.columns = ['name_ratio', 'name_token_set', 'name_token_sort', 'name_jaro']
    
    address_sims = df.apply(lambda row: calculate_similarity_features(row, 'address_1', 'address_2'), axis=1, result_type='expand')
    address_sims.columns = ['addr_ratio', 'addr_token_set', 'addr_token_sort', 'addr_jaro']
    
    # Exact signals
    df['country_match'] = (df['country_1'] == df['country_2']).astype(int)
    
    # Structure
    df['name_len_1'] = df['name_1'].fillna('').astype(str).apply(len)
    df['name_len_2'] = df['name_2'].fillna('').astype(str).apply(len)
    df['addr_len_1'] = df['address_1'].fillna('').astype(str).apply(len)
    df['addr_len_2'] = df['address_2'].fillna('').astype(str).apply(len)
    
    # Combine features
    features = pd.concat([df[['source1_entity_id', f'{target_prefix}_entity_id', 'country_match', 
                              'name_len_1', 'name_len_2', 'addr_len_1', 'addr_len_2', 'country_1']], 
                          name_sims, address_sims], axis=1)
    
    return features
