import os, sys, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
sys.path.insert(0, ROOT)
from src.models.pretrained import WeakMultiTaskModel

class MimicDataset(Dataset):
    def __init__(self, cls_file, extra_file, parquet_file):
        self.bert_emb = np.load(cls_file).astype(np.float32)
        extra       = np.load(extra_file).astype(np.float32)
        df          = pd.read_parquet(parquet_file)
        N = min(len(self.bert_emb), extra.shape[0], len(df))
        self.bert_emb = self.bert_emb[:N]
        extra         = extra[:N]
        df            = df.iloc[:N].reset_index(drop=True)
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls_embeddings", required=True)
    p.add_argument("--extra_feats",    required=True)
    p.add_argument("--parquet",        required=True)
    p.add_argument("--model_path",     required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = MimicDataset(args.cls_embeddings, args.extra_feats, args.parquet)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    emb_dim = ds.bert_emb.shape[1]
    num_dim = ds.numeric_feats.shape[1]
    ts_dim  = ds.ts_feats.shape[1]
    model = WeakMultiTaskModel(emb_dim, num_dim, ts_dim).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true = {"mortality":[], "long_los":[]}
    y_score= {"mortality":[], "long_los":[]}
    with torch.no_grad():
        for emb, num, ts, m, l in dl:
            emb, num, ts = emb.to(device), num.to(device), ts.to(device)
            pm, pl = model(emb, num, ts)
            y_true["mortality"].extend(m.tolist())
            y_true["long_los"].extend(l.tolist())
            y_score["mortality"].extend(pm.cpu().tolist())
            y_score["long_los"].extend(pl.cpu().tolist())

    print("\n=== Eval ===")
    for t in ("mortality","long_los"):
        print(f"{t:12s}  ROC‐AUC: {roc_auc_score(y_true[t], y_score[t]):.4f}   "
              f"PR‐AUC: {average_precision_score(y_true[t], y_score[t]):.4f}")

if __name__=="__main__":
    main()
