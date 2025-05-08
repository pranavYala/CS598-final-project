import os, argparse, sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
sys.path.insert(0, ROOT)

from src.models.pretrained import MaskedImputationPretrainer, MultiTaskPretrainer
from src.data.multitask_dataset import MultiTaskMimicDataset
from src.models.encoder     import GRUEncoder


def main(args):
    ds = MultiTaskMimicDataset(args.emb, args.parquet)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 1) Masked‐Imputation
    encoder = GRUEncoder(input_dim=ds.x.shape[1])
    mi      = MaskedImputationPretrainer(encoder, mask_prob=args.mask_prob).to(args.device)
    opt1    = torch.optim.Adam(mi.parameters(), lr=args.mi_lr)
    for e in range(args.mi_epochs):
        total=0
        for x,_ in dl:
            x = x.to(args.device)
            out = mi(x)
            loss= mi.loss(out)
            opt1.zero_grad(); loss.backward(); opt1.step()
            total+= loss.item()
        print(f"[MI] Epoch {e} – loss {total/len(dl):.4f}")
    torch.save(mi.encoder.state_dict(), os.path.join(args.ckpt_dir, 'encoder_mi.pt'))

    # 2) Full Multi‐Task
    # build your 10 heads exactly as authors: dims match each label
    H = encoder.output_dim
    heads = {
      'MOR': nn.Linear(H,1),
      'CMO': nn.Linear(H,1),
      'DNR': nn.Linear(H,1),
      'DIS': nn.Linear(H,1),
      'LOS': nn.Linear(H,1),
      'REA': nn.Linear(H,1),
      'ICD': nn.Linear(H, ds.labels['ICD'].shape[1]),
      'ACU': nn.Linear(H, ds.labels['ACU'].max()+1),
      'WBM': nn.Linear(H, ds.labels['WBM'].shape[1]),
      'FTS': nn.LSTMDecoder(H, embed_dim, ...),  # see fts_decoder in authors' code
    }
    mt = MultiTaskPretrainer(encoder, heads).to(args.device)
    opt2 = torch.optim.Adam(mt.parameters(), lr=args.mt_lr)
    for e in range(args.mt_epochs):
        total=0
        for x,labels in dl:
            x = x.to(args.device)
            outs = mt(x)
            loss = mt.loss(outs, labels)
            opt2.zero_grad(); loss.backward(); opt2.step()
            total+= loss.item()
        print(f"[MT] Epoch {e} – loss {total/len(dl):.4f}")
        torch.save(mt.state_dict(), os.path.join(args.ckpt_dir, f'mt_epoch{e}.pt'))

    # 3) Task-Omitted (M¬t): repeat MT but passing exclude_task=…
    for t in heads:
        mt_i = MultiTaskPretrainer(encoder, heads).to(args.device)
        opt3 = torch.optim.Adam(mt_i.parameters(), lr=args.mt_lr)
        for e in range(args.mt_epochs):
            for x,labels in dl:
                outs = mt_i(x)
                loss = mt_i.loss(outs, labels, exclude_task=t)
                opt3.zero_grad(); loss.backward(); opt3.step()
        torch.save(mt_i.state_dict(), os.path.join(args.ckpt_dir, f'mt_no_{t}.pt'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--emb',      default='src/data/preprocessed/mimic_embeddings_with_timeseries.npy')
    p.add_argument('--parquet',  default='src/data/preprocessed/mimic_examples.parquet')
    p.add_argument('--ckpt_dir', default='models/pretrain', help="where to save checkpoints")
    p.add_argument('--mask_prob',type=float, default=0.15)
    p.add_argument('--mi_lr',    type=float, default=1e-3)
    p.add_argument('--mt_lr',    type=float, default=1e-4)
    p.add_argument('--mi_epochs',type=int,   default=50)
    p.add_argument('--mt_epochs',type=int,   default=50)
    p.add_argument('--device',   default='cuda')
    args=p.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main(args)
