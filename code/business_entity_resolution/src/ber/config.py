"""All pipeline constants in one place."""
SEED = 42
NORM_LEVEL = 4                      # L4: NFKC cleanup + Indic transliteration (learned dict) + canonical forms

# training-S1 partition by a fixed random priority (same draw as the experiment lab, so nothing leaks):
TRAIN_A = (0.00, 0.15)              # fits the cheap and full LightGBM matchers
TRAIN_B = (0.15, 0.30)              # out-of-sample for A: fits stage 2, tunes thresholds, reports F0.5
FT_RANGE = (0.50, 0.53)             # S1 entities whose pairs fine-tune the encoder
# The test pool holds 5.75 records per S1 vs 4.68 in train (~40% ownerless vs 26%): as if ~19% of the S1 entities
# were removed while their S2/S3 copies stayed. Training reproduces this by hiding this S1 slice from the search.
TRAIN_HIDDEN_S1 = (0.30, 0.49)
DICT_MIN_PRIO = 0.60                # S1 entities whose pairs learn the native-script -> Latin dictionary
DICT_PAIRS = 600_000

# encoder (MIT licence, 118M parameters)
EMB_BASE = "intfloat/multilingual-e5-small"
EMB_PREFIX = "query: "
EMB_MAXLEN = 64
FT_PAIRS = 300_000
FT_BATCH = 128
FT_LR = 2e-5

# blocking: exact GPU nearest neighbours inside each country
# A smaller candidate set ranks higher in the final evaluation. Widening to 20/5 (v5) raised blocking recall
# 0.9921 -> 0.9954 but left held-out F0.5 flat (0.9851 vs 0.9852) and grew the final set 6.09 -> 7.62 per S1.
K_FWD = 10                          # S1 -> pool top-k
M_REV = 3                           # pool -> S1 top-m
# index behind the search: "ivf" = inverted-file k-means lists, each query scans only its IVF_NPROBE nearest lists
# (sublinear, the design that scales to billions); "exact" = scan the whole country (the v1-v5 setting)
BLOCK_INDEX = "exact"
IVF_LIST_SIZE = 1024                # average records per list -> n_lists = records / IVF_LIST_SIZE
IVF_NPROBE = 32                     # lists scanned per query

CTX = ["cos", "fwd_rank", "rev_rank", "gap_q", "rev_margin", "mutual", "n_cand_q", "is_s3"]
NAME = ["name_tri_j", "name_tok_j", "core_tok_j", "core_contain", "core_exact", "nospace_eq", "nospace_contain",
        "skel_j", "name_jw", "name_tsr", "name_tsort", "name_partial", "core_idf_q", "core_idf_c", "name_ntok_q",
        "name_ntok_c", "c_indic"]
ADDR = ["addr_tri_j", "addr_tok_j", "addr_contain", "addr_idf_q", "addr_idf_c", "addr_tsr", "addr_empty_c",
        "addr_ntok_q", "addr_ntok_c"]
NUM = ["num_n_q", "num_n_c", "num_j", "num_share", "num_contain_c", "num_conflict", "first_eq", "zip_eq"]
PAIR_FEATURES = NAME + ADDR + NUM
PAIR_V2 = ["name_tri_idfcos", "name_tri_idfcont", "addr_tri_idfcos", "addr_tri_idfcont", "core_info_q", "core_info_c",
           "shared_max_idf", "unshared_max_idf_q", "unshared_max_idf_c", "addr_unshared_max_idf_q",
           "addr_unshared_max_idf_c", "name_ratio", "name_jw_full"]
CHEAP = CTX + NUM + ["core_tok_j", "name_tok_j", "skel_j", "core_idf_q", "addr_tok_j", "addr_contain"]
FULL = CTX + NAME + ADDR + NUM

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED)
LGB_STAGE2 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
                  verbose=-1, seed=SEED)
LGB_ROUNDS = 2000

USE_V2 = True                       # V2 pair features in the full matcher
USE_V3 = True                       # V3 features: name uniqueness + fuzzy house numbers
USE_S2X = True                      # embedding-cluster + relative features in stage 2
LGB_FULL = LGB_PARAMS               # full-matcher parameters
RULES_TRY = ("threshold", "top1_plus", "relative", "expected_f")   # decision rules compared on B folds 0-2

CHEAP_THRESHOLDS = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02)
CHEAP_MAX_RECALL_LOSS = 0.001       # largest cheap threshold losing at most this much pair recall on B
