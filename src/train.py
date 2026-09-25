import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
import joblib

def train_model(df_features, labels, groups, cat_features=['country_1']):
    """
    Train CatBoost model with GroupKFold based on source1_entity_id.
    """
    df_features = df_features.fillna(-1)
    
    # Ensure categorical features are string
    for cat_feat in cat_features:
        if cat_feat in df_features.columns:
            df_features[cat_feat] = df_features[cat_feat].astype(str)
            
    X = df_features.drop(columns=['source1_entity_id', 'S2_entity_id', 'S3_entity_id', 'target_entity_id'], errors='ignore')
    y = labels
    
    # Identify categorical feature indices
    cat_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
    
    gkf = GroupKFold(n_splits=5)
    models = []
    
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            eval_metric='Logloss',
            random_seed=42,
            verbose=100
        )
        
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        models.append(model)
        
    return models

if __name__ == '__main__':
    # This acts as an entrypoint for training if run standalone
    pass
