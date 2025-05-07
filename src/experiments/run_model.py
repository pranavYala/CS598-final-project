#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import model class
from src.models.pretrained import WeakMultiTaskModel

class MimicDataset(Dataset):
    def __init__(self, embeddings_file, parquet_file):
        # 1) load combined BERT + timeseries embeddings
        self.embeddings = np.load(embeddings_file).astype(np.float32)
        # 2) load the DataFrame with raw ts + static + labels
        df = pd.read_parquet(parquet_file)
        # numeric (chart_mean_val, lab_mean_val)
        self.numeric_feats = df[['chart_mean_val','lab_mean_val']].to_numpy(dtype=np.float32)
        # time-series (HR, SBP, DBP - each length-24 arrays)
        hr  = np.stack(df['heart_rate_ts'].values).astype(np.float32)
        sbp = np.stack(df['sbp_ts'].values).astype(np.float32)
        dbp = np.stack(df['dbp_ts'].values).astype(np.float32)
        # concatenate into one vector per patient: shape (N, 24*3)
        self.time_series_feats = np.concatenate([hr, sbp, dbp], axis=1)
        # binary labels
        self.labels = {
            'mortality': df['mortality'].to_numpy(dtype=np.float32),
            'long_los':  df['long_los'].to_numpy(dtype=np.float32),
        }
        assert len(self.embeddings) == len(df) == len(self.numeric_feats) == len(self.time_series_feats)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb    = torch.from_numpy(self.embeddings[idx])
        feats  = torch.from_numpy(self.numeric_feats[idx])
        ts     = torch.from_numpy(self.time_series_feats[idx])
        mort   = torch.tensor(self.labels['mortality'][idx])
        los    = torch.tensor(self.labels['long_los'][idx])
        return emb, feats, ts, mort, los

def train(args):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # dataset + dataloader
    ds = MimicDataset(args.embeddings, args.parquet)
    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True)

    # model
    emb_dim = ds.embeddings.shape[1]
    num_num = ds.numeric_feats.shape[1]
    num_ts  = ds.time_series_feats.shape[1]

    model = WeakMultiTaskModel(
        emb_dim=emb_dim,
        num_numeric_feats=num_num,
        num_time_series_feats=num_ts
    ).to(device)

    # loss & optimizer
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        running, run_m, run_l = 0.0, 0.0, 0.0
        for emb, feats, ts, mort, los in loader:
            emb, feats, ts = emb.to(device), feats.to(device), ts.to(device)
            mort, los      = mort.to(device), los.to(device)

            optimizer.zero_grad()
            p_mort, p_los = model(emb, feats, ts)

            l_m = loss_fn(p_mort, mort)
            l_l = loss_fn(p_los,  los)
            loss = l_m + l_l
            loss.backward()
            optimizer.step()

            running += loss.item()
            run_m   += l_m.item()
            run_l   += l_l.item()

        n_batches = len(loader)
        print(f"Epoch {epoch}/{args.epochs} — "
              f"Total: {running/n_batches:.4f}  "
              f"MOR: {run_m/n_batches:.4f}  "
              f"LOS: {run_l/n_batches:.4f}")

    # save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

def parse_args():
    p = argparse.ArgumentParser(description="Train weakly-supervised multitask on MIMIC-III")
    p.add_argument('--embeddings', type=str,
                   default='data/preprocessed/mimic_embeddings_with_timeseries.npy',
                   help="path to .npy file of BERT+TS embeddings")
    p.add_argument('--parquet', type=str,
                   default='data/preprocessed/mimic_examples.parquet',
                   help="path to parquet with time-series and labels")
    p.add_argument('--output', type=str,
                   default='data/preprocessed/weak_pretrained_model.pt',
                   help="where to save trained model")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=10)
    p.add_argument('--lr',         type=float, default=1e-5)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--cuda',       action='store_true',
                   help="use CUDA if available")
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
