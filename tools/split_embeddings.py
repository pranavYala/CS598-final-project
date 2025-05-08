import numpy as np

all_feats = np.load("src/data/preprocessed/mimic_embeddings_with_timeseries.npy")  # (N, 842)
# first 768 dims are BERT-CLS, last 74 are TS (72) + static (2)
bert = all_feats[:, :768]
extra = all_feats[:, 768:]        # shape (N,74)

np.save("src/data/preprocessed/mimic_cls_embeddings.npy", bert)
np.save("src/data/preprocessed/mimic_additional_feats.npy", extra)
print("Saved mimic_cls_embeddings.npy and mimic_additional_feats.npy")
