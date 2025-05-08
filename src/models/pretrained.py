import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch.nn.functional as F

# Constants
preprocessed_dir = '../data/preprocessed'
embedding_file   = 'mimic_embeddings_with_timeseries.npy'
parquet     = 'mimic_examples.parquet'
output_model     = 'weak_pretrained_model.pt'

# Custom dataset
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

# simple multi-task model
class WeakMultiTaskModel(nn.Module):
    def __init__(self, emb_dim, num_numeric_feats, num_time_series_feats):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(emb_dim + num_numeric_feats + num_time_series_feats, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5) 
        )
        self.out_mort = nn.Linear(128, 1)
        self.out_los  = nn.Linear(128, 1)

    def forward(self, emb, feats, ts_feats):
        x      = torch.cat([emb, feats, ts_feats], dim=1)
        hidden = self.shared(x)
        return self.out_mort(hidden).squeeze(1), self.out_los(hidden).squeeze(1)
    
class MaskedImputationPretrainer(nn.Module):
    # Encoder + decoder for masked imputation

    def __init__(self, encoder, mask_prob=0.15):
        super().__init__()
        self.encoder    = encoder
        # decoder maps encoder.output_dim → input_dim (i.e. #features per time‐step)
        self.decoder    = nn.Linear(encoder.output_dim, encoder.input_dim)
        self.mask_prob  = mask_prob

    def forward(self, x):
        # x: [B, T•F] (flattened TS+static), or you can reshape inside encoder
        mask = (torch.rand_like(x) < self.mask_prob).float()
        x_masked = x * (1 - mask)
        repr = self.encoder(x_masked)
        pred = self.decoder(repr)
        return pred, x, mask

    def loss(self, forward_out):
        pred, orig, mask = forward_out
        # only penalize masked positions
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return F.mse_loss(pred * mask, orig * mask)

class MultiTaskPretrainer(nn.Module):
    # Pretrained encoder + task heads

    def __init__(self, encoder, task_heads: dict):
        super().__init__()
        self.encoder    = encoder
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x):
        h = self.encoder(x)                          # [B, H]
        return {t: head(h).squeeze(1) for t, head in self.task_heads.items()}

    def loss(self, outs: dict, labels: dict, exclude_task=None):
        total = 0.0
        for t, pred in outs.items():
            if t == exclude_task: continue
            target = labels[t].to(pred.device)
            total += F.binary_cross_entropy_with_logits(pred, target)
        return total

# Training function
def train_weak_multitask_model():
    # Load embeddings + DataFrame
    embeddings = np.load(os.path.join(preprocessed_dir, embedding_file)).astype(np.float32)
    df = pd.read_parquet(os.path.join(preprocessed_dir, parquet))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # Training loop
    for epoch in range(1, 8):
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
        print(f"Epoch {epoch:2d}/7 — Total Loss: {avg_loss:.4f} | Mort Loss: {avg_loss_mort:.4f} | LOS Loss: {avg_loss_los:.4f}")

    # Save model
    out_path = os.path.join(preprocessed_dir, output_model)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")

if __name__ == '__main__':
    train_weak_multitask_model()
