import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import pandas as pd

# === Constants ===
PREPROCESSED_DIR = '../data/preprocessed'
BERT_MODEL_NAME = 'bert-base-uncased'  # Can be swapped out

def load_preprocessed_data():
    """
    Load the preprocessed Parquet file.
    """
    path = os.path.join(PREPROCESSED_DIR, 'mimic_examples.parquet')
    print(f'Loading preprocessed data from {path}...')
    df = pd.read_parquet(path)
    return df

def prepare_tensors(df):
    """
    Convert BERT inputs into torch tensors.
    """
    all_input_ids = torch.tensor(df['input_ids'].tolist(), dtype=torch.long)
    all_attention_mask = torch.tensor(df['attention_mask'].tolist(), dtype=torch.long)
    return all_input_ids, all_attention_mask

def generate_embeddings(input_ids, attention_mask, batch_size=8, use_gpu=False):
    """
    Run BERT on batches to extract [CLS] embeddings.
    """
    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    print(f'Using device: {device}')

    model = BertModel.from_pretrained(BERT_MODEL_NAME)
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start_idx in range(0, len(input_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(input_ids))
            batch_input_ids = input_ids[start_idx:end_idx].to(device)
            batch_attention_mask = attention_mask[start_idx:end_idx].to(device)

            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            pooled = outputs.pooler_output  # shape: (B, hidden_size)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)  # shape: (N, hidden_size)

def extract_additional_features(df):
    """
    Extract flattened time series (HR, SBP, DBP) and static features (chart/lab).
    """
    hr_series = df['heart_rate_ts'].apply(lambda x: np.array(x, dtype=np.float32))
    sbp_series = df['sbp_ts'].apply(lambda x: np.array(x, dtype=np.float32))
    dbp_series = df['dbp_ts'].apply(lambda x: np.array(x, dtype=np.float32))

    # Flatten and stack: (N, 24*3)
    time_series_features = np.stack([
        np.concatenate([hr, sbp, dbp])
        for hr, sbp, dbp in zip(hr_series, sbp_series, dbp_series)
    ])

    # Optionally add chart/lab summary stats (static features)
    static_features = df[['chart_mean_val', 'lab_mean_val']].to_numpy(dtype=np.float32)

    # Combine: (N, 24*3 + 2)
    combined_features = np.hstack([time_series_features, static_features])
    print(f'Extracted additional features with shape: {combined_features.shape}')
    return combined_features

def main():
    df = load_preprocessed_data()

    # === BERT ===
    input_ids, attention_mask = prepare_tensors(df)
    cls_embeddings = generate_embeddings(input_ids, attention_mask, batch_size=8, use_gpu=True)
    print(f'BERT embeddings shape: {cls_embeddings.shape}')  # (N, hidden)

    # === Time series + chart/lab ===
    additional_feats = extract_additional_features(df)

    # === Combine final embeddings ===
    final_embeddings = np.hstack([cls_embeddings, additional_feats])
    print(f'Final embedding shape: {final_embeddings.shape}')  # (N, hidden + 24*3 + 2)

    # Save
    out_path = os.path.join(PREPROCESSED_DIR, 'mimic_embeddings_with_timeseries.npy')
    np.save(out_path, final_embeddings)
    print(f'Saved combined embeddings to {out_path}')

if __name__ == '__main__':
    main()
