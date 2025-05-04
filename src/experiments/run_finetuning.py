import torch, numpy as np
from torch.utils.data import DataLoader, random_split
from src.models.encoder import GRUEncoder
from src.models.finetune import FineTuner
import torch.nn as nn

def get_dataloaders(task, batch_size=64):
    # sketch: load same NPZDataset but select only `data` and `labels_{task}`
    ds = NPZDataset([...])
    n = len(ds)
    n_train = int(n*0.8)
    train, val = random_split(ds, [n_train, n-n_train])
    return DataLoader(train, batch_size=batch_size, shuffle=True), \
           DataLoader(val,   batch_size=batch_size)

def main():
    tasks = ['MOR','CMO','DNR','DIS','ICD','LOS','REA','ACU','WBM','FTS']
    for pt in ['mi','mt','none']:
        # load encoder
        enc = GRUEncoder(input_dim=...).cuda()
        if pt=='mi':
            enc.load_state_dict(torch.load('models/mi_encoder.pt'))
        elif pt=='mt':
            enc.load_state_dict(torch.load('models/mt_encoder.pt'))
        for task in tasks:
            # build head
            if task in ['ICD','ACU','DIS']:
                out_dim = N_CLASSES[task]
                head = nn.Linear(enc.output_dim, out_dim)
            else:
                head = nn.Linear(enc.output_dim, 1)
            ft = FineTuner(enc, head, freeze_encoder=False)
            tr,va = get_dataloaders(task)
            ft.train(tr, va, epochs=20, lr=1e-3)
            torch.save(ft.model.state_dict(), f'models/{pt}_{task}_ft.pt')

if __name__=='__main__':
    main()
