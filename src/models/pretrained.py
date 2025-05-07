import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# --- Constants ---
PREPROCESSED_DIR = '../data/preprocessed'
EMBEDDING_FILE   = 'mimic_embeddings_with_timeseries.npy'
PARQUET_FILE     = 'mimic_examples.parquet'
MODEL_OUTPUT     = 'weak_pretrained_model.pt'

# --- Custom Dataset ---
class MimicPretrainDataset(Dataset):
    def __init__(self, embeddings, numeric_feats, time_series_feats, labels):
        self.embeddings        = embeddings
        self.numeric_feats     = numeric_feats
        self.time_series_feats = time_series_feats
        self.labels            = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb     = torch.from_numpy(self.embeddings[idx])        # float32
        feats   = torch.from_numpy(self.numeric_feats[idx])     # float32
        ts_feats= torch.from_numpy(self.time_series_feats[idx]) # float32
        mort    = torch.tensor(self.labels['mortality'][idx])   # float32
        los     = torch.tensor(self.labels['long_los'][idx])    # float32
        return emb, feats, ts_feats, mort, los

# --- Simple Multi-task Model ---
class WeakMultiTaskModel(nn.Module):
    def __init__(self, emb_dim, num_numeric_feats, num_time_series_feats):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(emb_dim + num_numeric_feats + num_time_series_feats, 128),
            nn.ReLU()
        )
        self.out_mort = nn.Linear(128, 1)
        self.out_los  = nn.Linear(128, 1)

    def forward(self, emb, feats, ts_feats):
        x      = torch.cat([emb, feats, ts_feats], dim=1)
        hidden = self.shared(x)
        return self.out_mort(hidden).squeeze(1), self.out_los(hidden).squeeze(1)

# --- Training Function ---
def train_weak_multitask_model():
    # Load embeddings + DataFrame
    embeddings = np.load(os.path.join(PREPROCESSED_DIR, EMBEDDING_FILE)).astype(np.float32)
    df = pd.read_parquet(os.path.join(PREPROCESSED_DIR, PARQUET_FILE))

    # Numeric features
    numeric_feats = df[['chart_mean_val', 'lab_mean_val']].values.astype(np.float32)

    # Time-series features
    hr  = np.stack(df['heart_rate_ts'].values).astype(np.float32)
    sbp = np.stack(df['sbp_ts'].values).astype(np.float32)
    dbp = np.stack(df['dbp_ts'].values).astype(np.float32)
    time_series_feats = np.concatenate([hr, sbp, dbp], axis=1)

    # Labels
    labels = {
        'mortality': df['mortality'].values.astype(np.float32),
        'long_los':  df['long_los'].values.astype(np.float32),
    }

    # Dataset and DataLoader
    dataset = MimicPretrainDataset(embeddings, numeric_feats, time_series_feats, labels)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Model, loss, optimizer
    model = WeakMultiTaskModel(
        emb_dim=embeddings.shape[1],
        num_numeric_feats=numeric_feats.shape[1],
        num_time_series_feats=time_series_feats.shape[1],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        total_loss_mort = 0.0
        total_loss_los  = 0.0

        for emb, feats, ts_feats, mort, los in loader:
            emb, feats, ts_feats = emb.to(device), feats.to(device), ts_feats.to(device)
            mort, los = mort.to(device), los.to(device)

            optimizer.zero_grad()
            pred_mort, pred_los = model(emb, feats, ts_feats)

            loss_mort = loss_fn(pred_mort, mort)
            loss_los  = loss_fn(pred_los, los)
            loss = loss_mort + loss_los

            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            total_loss_mort += loss_mort.item()
            total_loss_los  += loss_los.item()

        avg_loss      = total_loss / len(loader)
        avg_loss_mort = total_loss_mort / len(loader)
        avg_loss_los  = total_loss_los / len(loader)
        print(f"Epoch {epoch:2d}/10 — Total Loss: {avg_loss:.4f} | Mort Loss: {avg_loss_mort:.4f} | LOS Loss: {avg_loss_los:.4f}")

    # Save model
    out_path = os.path.join(PREPROCESSED_DIR, MODEL_OUTPUT)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")

if __name__ == '__main__':
    train_weak_multitask_model()
