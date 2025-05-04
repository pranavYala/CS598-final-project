import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FineTuner:
    def __init__(self, encoder: nn.Module, head: nn.Module, freeze_encoder: bool = False):
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
        self.model = nn.Sequential(encoder, head)

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 20,
              lr: float = 1e-3,
              device: str = 'cuda'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.model.to(device)
        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss=0
            for x,y in train_loader:
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                if y.dim()==1 or y.shape[1]==1:
                    loss = F.binary_cross_entropy_with_logits(out.squeeze(), y.float())
                else:
                    loss = F.cross_entropy(out, y.argmax(dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch} train loss {total_loss/len(train_loader):.4f}')
            self.validate(val_loader, device)

    def validate(self, loader: DataLoader, device='cuda'):
        from sklearn.metrics import roc_auc_score, accuracy_score
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                logits = self.model(x)
                probs = torch.sigmoid(logits) if logits.shape[-1]==1 else F.softmax(logits, dim=-1)
                ys.append(y.cpu().numpy())
                ps.append(probs.cpu().numpy())
        y_true = np.concatenate(ys,0)
        y_pred = np.concatenate(ps,0)
        # binary/multilabel vs multiclass
        if y_true.ndim==1 or y_true.shape[1]==1:
            auc = roc_auc_score(y_true, y_pred.squeeze())
            acc = accuracy_score(y_true, y_pred.squeeze()>0.5)
            print(f'  VAL  AUC={auc:.4f} ACC={acc:.4f}')
        else:
            # compute per-class AUC or overall accuracy
            preds = y_pred.argmax(axis=1)
            true = y_true.argmax(axis=1)
            acc = accuracy_score(true, preds)
            print(f'  VAL  Acc={acc:.4f}')
