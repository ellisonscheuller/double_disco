import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from embedding.dataloader import PFCandsDataset, PUPPIDataset
from embedding.utils.data_utils import delta_r_from_normalized


class EarlyStopping:
    """
    Simple early stopping on a monitored value (default: minimize 'loss').
    mode='min' or 'max'
    """
    def __init__(self, patience=20, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False  # do not stop

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad > self.patience

#nurd critic model
class NuRDCritic(nn.Module):
    """
    Binary classifier: distinguish joint vs shuffled nuisance.
    Input: [embeddings (z_dim), ae_score (1), label (1)] 
    """
    def __init__(self, z_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 2, hidden),  # +ae(1) +label(1)
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_train_val_split(features, y, val_size=0.10, random_state=42, y_are_labels=True):
    """
    Stratified split of event tensors into train/val.
    features: [E, N, F] (normalized)
    y (labels):   [E]
    """
    idx = torch.arange(features.shape[0])
    idx_tr, idx_val = train_test_split(
        idx.cpu().numpy(),
        test_size=val_size,
        random_state=random_state,
        stratify=y.cpu().numpy() if y_are_labels else None
    )
    idx_tr = torch.tensor(idx_tr, dtype=torch.long, device=features.device)
    idx_val = torch.tensor(idx_val, dtype=torch.long, device=features.device)

    X_tr = features.index_select(0, idx_tr)
    y_tr = y.index_select(0, idx_tr)
    X_val = features.index_select(0, idx_val)
    y_val = y.index_select(0, idx_val)
    return X_tr, y_tr, X_val, y_val, idx_tr, idx_val


def build_train_val_loaders(
    X_tr, y_tr, X_val, y_val, device, batch_size=2048, pfcands=False,
    ae_scores_tr=None, ae_scores_val=None,  # NEW: named to match your main script
):
    """
    Builds DataLoaders. If AE scores are provided, the dataset returns 4-tuple:
        (x, mask, label, ae_score)
    else returns the usual 3-tuple:
        (x, mask, label)
    """
    if pfcands:
        ds_tr = PFCandsDataset(X_tr,  y_tr, device, ae_scores=ae_scores_tr)   # NEW
        ds_val = PFCandsDataset(X_val, y_val, device, ae_scores=ae_scores_val) # NEW
    else:
        ds_tr = PUPPIDataset(X_tr,  y_tr, device=device, ae_scores=ae_scores_tr)   # NEW
        ds_val = PUPPIDataset(X_val, y_val, device=device, ae_scores=ae_scores_val) # NEW

    train_loader = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_epoch(
    encoder, projector, classifier,
    ce_loss_fn, contrastive_loss,
    train_loader, norm_constants, device,
    optimizer, scheduler=None, contrastive_weight=0.05,
    pairwise=False,

    #nurd params
    nurd_critic=None,
    nurd_critic_optimizer=None,
    nurd_lambda=0.0,
    nurd_critic_steps=1,
):
    """
    One training epoch. Returns averaged metrics.
    If AE scores are present in the loader and nurd_lambda>0, adds NuRD penalty.
    """
    encoder.train(); projector.train(); classifier.train()
    if nurd_critic is not None:
        nurd_critic.train()  # NEW

    total_loss = total_nce = total_ce = 0.0
    total_info = 0.0  # NEW
    correct = count = 0

    for batch in train_loader:
        if len(batch) == 3:
            x, mask, labels = batch
            ae = None
        else:
            x, mask, labels, ae = batch 

        x = x.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        if ae is not None:
            ae = ae.to(device).float().view(-1, 1)  # (B,1)

        # prepend CLS bit
        mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
            mask.bool()
        ], dim=1)

        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        latent = encoder(x, delta_r, mask)
        embeddings = F.normalize(projector(latent), dim=1)

        loss_nce = contrastive_loss(embeddings, labels)
        logits   = classifier(embeddings)
        loss_ce  = ce_loss_fn(logits, labels)

        loss = contrastive_weight * loss_nce + loss_ce

        #NuRD decorrelation
        info_loss = torch.tensor(0.0, device=device)
        if (
            (nurd_critic is not None)
            and (nurd_critic_optimizer is not None)
            and (nurd_lambda > 0.0)
            and (ae is not None)
        ):
            # Train critic on detached z
            z_det = embeddings.detach()
            y_float = labels.float().view(-1, 1)

            for _ in range(nurd_critic_steps):
                ae_shuf = ae[torch.randperm(ae.size(0))]

                inp_joint = torch.cat([z_det, ae, y_float], dim=1)
                inp_shuf = torch.cat([z_det, ae_shuf, y_float], dim=1)
                inp = torch.cat([inp_joint, inp_shuf], dim=0)

                tgt = torch.cat([
                    torch.ones(z_det.size(0), dtype=torch.long, device=device),   # joint = 1
                    torch.zeros(z_det.size(0), dtype=torch.long, device=device),  # shuffled = 0
                ], dim=0)

                perm = torch.randperm(inp.size(0))
                inp = inp[perm]
                tgt = tgt[perm]

                out = nurd_critic(inp)
                loss_critic = F.cross_entropy(out, tgt)

                nurd_critic_optimizer.zero_grad()
                loss_critic.backward()
                nurd_critic_optimizer.step()

            # Compute info term on non detached embeddings to push encoder/projector
            out_joint = nurd_critic(torch.cat([embeddings, ae, y_float], dim=1))
            logp = F.log_softmax(out_joint, dim=1)
            info_loss = (logp[:, 1] - logp[:, 0]).mean()

            loss = loss + nurd_lambda * info_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_nce += loss_nce.item() * bs
        total_ce += loss_ce.item() * bs
        total_info += float(info_loss.detach().item()) * bs  # NEW
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += bs

    return {
        "loss": total_loss / count,
        "nce":  total_nce  / count,
        "ce":   total_ce   / count,
        "info": total_info / count,  # NEW
        "acc":  correct / count
    }


@torch.no_grad()
def validate_epoch(
    encoder, projector, classifier,
    ce_loss_fn, contrastive_loss,
    val_loader, norm_constants, device,
    contrastive_weight=0.05,
    pairwise=False,

    nurd_critic=None,
    nurd_lambda=0.0,
):
    """
    Validation pass (no grads). If critic+AE are present, logs info term too.
    """
    encoder.eval(); projector.eval(); classifier.eval()
    if nurd_critic is not None:
        nurd_critic.eval()  # NEW

    total_loss = total_nce = total_ce = 0.0
    total_info = 0.0  # NEW
    correct = count = 0

    for batch in val_loader:
        if len(batch) == 3:
            x, mask, labels = batch
            ae = None
        else:
            x, mask, labels, ae = batch  # NEW

        x = x.to(device); mask = mask.to(device); labels = labels.to(device)
        if ae is not None:
            ae = ae.to(device).float().view(-1, 1)

        mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
            mask.bool()
        ], dim=1)

        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        latent = encoder(x, delta_r, mask)
        embeddings = F.normalize(projector(latent), dim=1)

        loss_nce = contrastive_loss(embeddings, labels)
        logits = classifier(embeddings)
        loss_ce  = ce_loss_fn(logits, labels)

        info_loss = torch.tensor(0.0, device=device)
        if (nurd_critic is not None) and (nurd_lambda > 0.0) and (ae is not None):
            y_float = labels.float().view(-1, 1)
            out_joint = nurd_critic(torch.cat([embeddings, ae, y_float], dim=1))
            logp = F.log_softmax(out_joint, dim=1)
            info_loss = (logp[:, 1] - logp[:, 0]).mean()

        loss = contrastive_weight * loss_nce + loss_ce + nurd_lambda * info_loss

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_nce += loss_nce.item() * bs
        total_ce += loss_ce.item() * bs
        total_info += float(info_loss.detach().item()) * bs 
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += bs

    return {
        "loss": total_loss / count,
        "nce": total_nce  / count,
        "ce":  total_ce  / count,
        "info": total_info / count,  
        "acc":  correct / count
    }


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
