import os, sys, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# make sure `src/` is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
sys.path.insert(0, ROOT)
from src.models.pretrained import WeakMultiTaskModel

class MimicDataset(Dataset):
    def __init__(self, cls_file, extra_file, parquet_file):
        # 1) load 768-dim CLS embeddings
        self.bert_emb = np.load(cls_file).astype(np.float32)
        # 2) load 74-dim extra TS+static
        extra = np.load(extra_file).astype(np.float32)
        # 3) load labels + sanity-check
        df = pd.read_parquet(parquet_file)
        N = min(len(self.bert_emb), extra.shape[0], len(df))
        self.bert_emb = self.bert_emb[:N]
        extra         = extra[:N]
        df            = df.iloc[:N].reset_index(drop=True)

        # split extra → ts_feats (first 72 dims) and numeric (last 2 dims)
        self.ts_feats      = extra[:, :72]
        self.numeric_feats = extra[:, 72:]

        self.mortality = df['mortality'].to_numpy(dtype=np.float32)
        self.long_los  = df['long_los'].to_numpy(dtype=np.float32)

    def __len__(self): return len(self.bert_emb)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.bert_emb[idx]),
            torch.from_numpy(self.numeric_feats[idx]),
            torch.from_numpy(self.ts_feats[idx]),
            torch.tensor(self.mortality[idx]),
            torch.tensor(self.long_los[idx]),
        )

def train_and_evaluate(args):
    # device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # dataset + splits
    ds = MimicDataset(args.cls_embeddings, args.extra_feats, args.parquet)
    N = len(ds)
    idxs = np.arange(N)
    trainval, test_idx = train_test_split(idxs, test_size=args.test_frac, random_state=args.seed)
    train_idx, val_idx = train_test_split(trainval, test_size=args.val_frac, random_state=args.seed)
    print(f"N={N} → train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    def make_loader(idxs, shuffle):
        return DataLoader(
            Subset(ds, idxs), batch_size=args.batch_size,
            shuffle=shuffle, num_workers=args.num_workers, pin_memory=True
        )
    train_loader = make_loader(train_idx, True)
    val_loader   = make_loader(val_idx,   False)
    test_loader  = make_loader(test_idx,  False)

    # model
    emb_dim = ds.bert_emb.shape[1]  # 768
    num_dim = ds.numeric_feats.shape[1]  # 2
    ts_dim  = ds.ts_feats.shape[1]      # 72
    model = WeakMultiTaskModel(emb_dim, num_dim, ts_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for emb, num, ts, m, l in train_loader:
            emb, num, ts = emb.to(device), num.to(device), ts.to(device)
            m, l         = m.to(device),   l.to(device)

            optimizer.zero_grad()
            pm, pl = model(emb, num, ts)
            loss = loss_fn(pm, m) + loss_fn(pl, l)
            loss.backward()
            optimizer.step()
            running += loss.item() * emb.size(0)

        train_loss = running / len(train_loader.dataset)

        # validation
        model.eval()
        val_sum = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for emb, num, ts, m, l in val_loader:
                emb, num, ts = emb.to(device), num.to(device), ts.to(device)
                pm, pl = model(emb, num, ts)
                y_true.extend(m.cpu().tolist() + l.cpu().tolist())
                y_pred.extend(pm.cpu().tolist() + pl.cpu().tolist())
                val_sum += ( loss_fn(pm, m).item() + loss_fn(pl, l).item() ) * emb.size(0)
        val_loss = val_sum / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)

    # test
    print("→ loading best model and evaluating on test set")
    model.load_state_dict(torch.load(args.output, map_location=device))
    model.eval()
    y_true, y_pred = {'mortality':[], 'long_los':[]}, {'mortality':[], 'long_los':[]}
    test_loss = 0.0
    with torch.no_grad():
        for emb, num, ts, m, l in test_loader:
            emb, num, ts = emb.to(device), num.to(device), ts.to(device)
            pm, pl = model(emb, num, ts)
            test_loss += ( loss_fn(pm, m).item() + loss_fn(pl, l).item() ) * emb.size(0)
            y_true['mortality'].extend(m.cpu().tolist())
            y_true['long_los'].extend(l.cpu().tolist())
            y_pred['mortality'].extend(pm.cpu().tolist())
            y_pred['long_los'].extend(pl.cpu().tolist())
    test_loss /= len(test_loader.dataset)

    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"\n=== Test ===  loss={test_loss:.4f}")
    for t in ('mortality','long_los'):
        print(f"{t} ROC-AUC = {roc_auc_score(y_true[t], y_pred[t]):.4f}  PR-AUC = {average_precision_score(y_true[t], y_pred[t]):.4f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cls_embeddings",  required=True,
                   help="768-dim BERT-CLS .npy")
    p.add_argument("--extra_feats",     required=True,
                   help="74-dim TS+static .npy")
    p.add_argument("--parquet",         required=True)
    p.add_argument("--output",          required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cuda",        action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--test_frac",   type=float, default=0.2)
    p.add_argument("--val_frac",    type=float, default=0.1)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_and_evaluate(args)
