import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import gc

def get_top_k_sparse(sparse_matrix, query_matrix, k=50):
    """
    Returns top k indices and scores for each query.
    sparse_matrix: (N, vocab) - e.g. S2 or S3
    query_matrix: (M, vocab) - e.g. S1
    """
    # This might require batching if M and N are large
    batch_size = 10000
    M = query_matrix.shape[0]
    
    top_indices = np.zeros((M, k), dtype=np.int32)
    top_scores = np.zeros((M, k), dtype=np.float32)
    
    for start_idx in range(0, M, batch_size):
        end_idx = min(start_idx + batch_size, M)
        query_batch = query_matrix[start_idx:end_idx]
        
        # dot product: (batch, vocab) x (vocab, N) -> (batch, N)
        scores = query_batch.dot(sparse_matrix.T)
        
        # We need top k for each row.
        # Instead of sorting all N, we use argpartition
        if scores.shape[1] > k:
            # Using array because sparse rows don't support argpartition well
            scores_arr = scores.toarray()
            # argpartition on negative to get largest
            part_indices = np.argpartition(-scores_arr, k, axis=1)[:, :k]
            # sort the top k
            for i in range(end_idx - start_idx):
                idx = part_indices[i]
                sorted_idx = idx[np.argsort(-scores_arr[i, idx])]
                top_indices[start_idx + i] = sorted_idx
                top_scores[start_idx + i] = scores_arr[i, sorted_idx]
        else:
            scores_arr = scores.toarray()
            sorted_idx = np.argsort(-scores_arr, axis=1)
            # pad if less than k
            # (ignoring padding logic for now as N is usually >> k)
            top_indices[start_idx:end_idx, :scores.shape[1]] = sorted_idx
            for i in range(end_idx - start_idx):
                top_scores[start_idx + i, :scores.shape[1]] = scores_arr[i, sorted_idx[i]]
                
        del scores, query_batch
        gc.collect()
        
    return top_indices, top_scores

def generate_tfidf_candidates(df_s1, df_s2, col_name, prefix, k=20):
    """
    Generate candidates using character n-gram TF-IDF.
    df_s1: Source 1 dataframe (must have col_name)
    df_s2: Source 2/3 dataframe (must have col_name)
    prefix: Prefix for candidate type (e.g., 'S2')
    """
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_df=0.8)
    
    # Fit on all to have a common vocabulary
    corpus = pd.concat([df_s1[col_name], df_s2[col_name]]).fillna('')
    vectorizer.fit(corpus)
    
    s2_matrix = vectorizer.transform(df_s2[col_name].fillna(''))
    s1_matrix = vectorizer.transform(df_s1[col_name].fillna(''))
    
    indices, scores = get_top_k_sparse(s2_matrix, s1_matrix, k=k)
    
    # Construct candidate dataframe
    records = []
    s1_ids = df_s1['entity_id'].values
    s2_ids = df_s2['entity_id'].values
    
    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        for j in range(k):
            score = scores[i, j]
            if score > 0.1: # threshold to save space
                records.append({
                    'source1_entity_id': s1_id,
                    f'{prefix}_entity_id': s2_ids[indices[i, j]],
                    'score': score
                })
                
    return pd.DataFrame(records)

# Further retrieval strategies like exact overlap can be added here
