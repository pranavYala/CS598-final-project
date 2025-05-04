import torch
from torch.utils.data import DataLoader
from src.data.preprocess import extract_cohort_mimic
from src.models.encoder import GRUEncoder
from src.models.pretrained import MaskedImputationPretrainer, MultiTaskPretrainer

def main():
    # assume data is preprocessed into .npz files
    files = list('data/processed/'.__class__.__call__())  # list of npz paths
    # build a simple Dataset to load (sketch)
    class NPZDataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            z = np.load(self.files[i])
            return z['data'].astype(np.float32), \
                   { 'MOR': z['labels_MOR'], … }  # load labels per task

    ds = NPZDataset(files)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    encoder = GRUEncoder(input_dim=ds[0][0].shape[-1])
    # 1) Masked-Imputation
    mi = MaskedImputationPretrainer(encoder)
    optimizer = torch.optim.Adam(mi.parameters(), lr=1e-3)
    for epoch in range(10):
        for x,_ in dl:
            preds = mi(x)
            loss = mi.loss(preds)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"MI Pretrain Epoch {epoch} loss {loss.item():.4f}")
    torch.save(mi.encoder.state_dict(), 'models/mi_encoder.pt')

    # 2) Multi-Task
    # define one head per task (example binary heads)
    task_heads = {t: nn.Linear(encoder.output_dim, 1) for t in ['MOR','CMO','DNR',…]}
    mt = MultiTaskPretrainer(GRUEncoder(ds[0][0].shape[-1]), task_heads)
    optimizer = torch.optim.Adam(mt.parameters(), lr=1e-3)
    for epoch in range(10):
        for x,labels in dl:
            outs = mt(x)
            loss = mt.loss(outs, labels, exclude_task=None)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"MT Pretrain Epoch {epoch} loss {loss.item():.4f}")
    torch.save(mt.encoder.state_dict(), 'models/mt_encoder.pt')

if __name__=='__main__':
    main()

