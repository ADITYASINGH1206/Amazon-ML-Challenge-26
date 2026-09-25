import os
import pandas as pd
from pathlib import Path

# Paths
base_dir = Path("../../6ab10eb3b23ba_student_resource/student_resource/dataset").resolve()
train_dir = base_dir / "train"
test_dir = base_dir / "test"

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating mock datasets in {base_dir}")

# S1
s1_data = {
    "entity_id": ["S1-1", "S1-2", "S1-3"],
    "business_name": ["Acme Corp", "Tech Solutions", "French Bakery Ltd"],
    "business_address": ["123 Main St, New York", "456 Tech Ave, San Francisco", "10 Rue de la Paix, Paris"],
    "country": ["US", "US", "France"]
}
df_s1 = pd.DataFrame(s1_data)
df_s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
df_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

# S2
s2_data = {
    "entity_id": ["S2-1", "S2-2", "S2-3"],
    "business_name": ["Acme Corporation", "Tech Solutions Inc", "French Bakery"],
    "business_address": ["123 Main Street, NY", "456 Tech Avenue, SF", "10 Rue de la Paix, Paris, FR"],
    "country": ["US", "US", "France"]
}
df_s2 = pd.DataFrame(s2_data)
df_s2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
df_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

# S3
s3_data = {
    "entity_id": ["S3-1", "S3-2", "S3-3"],
    "business_name": ["Acme Corp.", "Tech Solutions", "Boulangerie Francaise"],
    "business_address": ["123 Main St., New York, NY", "456 Tech Ave, San Francisco, CA", "10 Rue de la Paix, Paris"],
    "country": ["US", "US", "France"]
}
df_s3 = pd.DataFrame(s3_data)
df_s3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
df_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

# Ground Truth
gt_data = {
    "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
    "matched_entity_ids": ["S2-1,S3-1", "S2-2,S3-2", "S2-3,S3-3"]
}
df_gt = pd.DataFrame(gt_data)
df_gt.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

print("Mock data creation complete!")
