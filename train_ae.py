import os
import random
import gc
import argparse
import numpy as np
import h5py as h5
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
from models.autoencoder import Autoencoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic

#setting a seed like in ae_legacy
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_h5_tree(h, prefix=""):
    for k in h.keys():
        item = h[k]
        if hasattr(item, "keys"):
            print(prefix + f"[GROUP] {k}")
            print_h5_tree(item, prefix + "  ")
        else:
            try:
                print(prefix + f"{k}: shape={item.shape}, dtype={item.dtype}")
            except Exception:
                print(prefix + f"{k}: <dataset>")

def fit_standard_scaler(X, eps=1e-8):
    mu  = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std)
    return mu, std

def transform_standard(X, mu, std):
    return (X - mu) / (std + 1e-8)

class PerSampleMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
    def forward(self, recon, target):
        per_feat = self.mse(recon, target)
        return per_feat.mean(dim=1)

def inference(ae, Xz, loss_fn, device, batch_size=4096):
    ae.eval()
    n = Xz.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, batch_size):
            i1 = min(i0 + batch_size, n)
            xb = torch.tensor(Xz[i0:i1], dtype=torch.float32, device=device)
            recon, _ = ae(xb)
            loss_b = loss_fn(recon, xb)
            out[i0:i1] = loss_b.detach().cpu().numpy()
    return out


def run_ae(config):
    set_seed(config.get("seed", 123))

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}", flush=True)

    print("Logging in to wandb...", flush=True)
    wandb.login(key="24d1d60ce26563c74d290d7b487cb104fc251271")
    wandb.init(
        project="AE vs. Contrastive ABCD",
        settings=wandb.Settings(_disable_stats=True),
        config=config,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    print(f"Run name: {wandb.run.name}", flush=True)

    #load .pt object tensors
    obj_train_path = config["obj_train_path"]
    obj_test_path = config["obj_test_path"]

    #event id tensors
    eid_train_path = config.get("eid_train_path", None)
    eid_test_path = config.get("eid_test_path", None)

    outdir = config.get("outdir", "outputs_ae_axis1")
    os.makedirs(outdir, exist_ok=True)

    print("Loading object .pt datasets...", flush=True)

    obj_train = torch.load(obj_train_path, map_location="cpu")  # (N, nobj, 5)
    obj_test  = torch.load(obj_test_path,  map_location="cpu")

    print("OBJ train shape:", tuple(obj_train.shape), flush=True)
    print("OBJ test  shape:", tuple(obj_test.shape),  flush=True)

    # strip off label column, keep 4 features: pt, eta, phi, type_id
    x_train = obj_train[:, :, :4].numpy().astype("float32")
    x_test  = obj_test[:, :, :4].numpy().astype("float32")

    # optional: load event ids so we can save them with scores
    eid_train = torch.load(eid_train_path, map_location="cpu").numpy() if eid_train_path else None  # (N,3)
    eid_test  = torch.load(eid_test_path,  map_location="cpu").numpy() if eid_test_path  else None

    def zero_out_padding(X):
        X = X.copy()
        pad = (X == 0.0).all(axis=-1)
        X[pad] = 0.0
        return X

    Xtr_raw = zero_out_padding(x_train)
    Xte_raw = zero_out_padding(x_test)

    def flatten(x):
        n, nobj, fdim = x.shape
        return x.reshape(n, nobj * fdim)

    X1_train_raw = flatten(Xtr_raw)
    X1_test_raw  = flatten(Xte_raw)

    #standardize
    mu1, std1 = fit_standard_scaler(X1_train_raw)
    X_train_z = transform_standard(X1_train_raw, mu1, std1)
    X_test_z  = transform_standard(X1_test_raw,  mu1, std1)

    #build AE
    feat = X_train_z.shape[1]
    reco_loss_fn = PerSampleMSE().to(device)

    ae_cfg = {
        "features": feat,
        "latent_dim": config["ae_latent"],
        "encoder_config": {"nodes": config["enc_nodes"]},
        "decoder_config": {"nodes": config["dec_nodes"] + [feat]},
        "alpha": config["alpha"],
    }

    ae = Autoencoder(ae_cfg).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=float(config["ae_lr"]))

    X1 = torch.tensor(X_train_z, dtype=torch.float32, device=device)

    #training AE
    print("Starting AE training...", flush=True)
    for epoch in range(config["epochs"]):
        perm = torch.randperm(len(X1))
        losses = []

        for i0 in range(0, len(X1), config["batch_size"]):
            idx = perm[i0 : i0 + config["batch_size"]]
            xb = X1[idx]

            recon, _ = ae(xb)
            loss = reco_loss_fn(recon, xb).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg = float(np.mean(losses))
        wandb.log({"epoch": epoch, "RecoLoss_AE": avg})
        print(f"Epoch {epoch}: Reco Loss: {avg:.6f}", flush=True)

    #save the AE
    ae_path = os.path.join(outdir, "ae_axis1.pth")
    torch.save(ae.state_dict(), ae_path)
    wandb.save(ae_path)
    print("Saved AE:", ae_path, flush=True)

    #compute AE scores
    ae_scores_bkg_train = inference(ae, X_train_z, reco_loss_fn, device, batch_size=4096)
    ae_scores_bkg_test  = inference(ae, X_test_z,  reco_loss_fn, device, batch_size=4096)

    #save scores
    np.save(os.path.join(outdir, "ae_scores_bkg_train.npy"), ae_scores_bkg_train.astype(np.float32))
    np.save(os.path.join(outdir, "ae_scores_bkg_test.npy"),  ae_scores_bkg_test.astype(np.float32))

    #save matching ids so you can align later with contrastive
    if eid_train is not None:
        np.save(os.path.join(outdir, "eventid_train.npy"), eid_train.astype(np.int64))
    if eid_test is not None:
        np.save(os.path.join(outdir, "eventid_test.npy"), eid_test.astype(np.int64))

    #wandb save
    wandb.save(os.path.join(outdir, "ae_scores_bkg_train.npy"))
    wandb.save(os.path.join(outdir, "ae_scores_bkg_test.npy"))
    if eid_train is not None:
        wandb.save(os.path.join(outdir, "eventid_train.npy"))
    if eid_test is not None:
        wandb.save(os.path.join(outdir, "eventid_test.npy"))

    wandb.finish()

    return {
        "ae": ae,
        "ae_path": ae_path,
        "outdir": outdir,
        "ae_scores_bkg_train": ae_scores_bkg_train,
        "ae_scores_bkg_test": ae_scores_bkg_test,
    }

#define AE config
cfg_ae = {
  "obj_train_path": "/uscms_data/d3/escheull/smcocktail_paired_1M/hlt_smcocktail_ae_obj_train.pt",
  "obj_test_path": "/uscms_data/d3/escheull/smcocktail_paired_1M/hlt_smcocktail_ae_obj_test.pt",
  "eid_train_path": "/uscms_data/d3/escheull/smcocktail_paired_1M/hlt_smcocktail_eventid_train.pt",
  "eid_test_path": "/uscms_data/d3/escheull/smcocktail_paired_1M/hlt_smcocktail_eventid_test.pt",

  "outdir": "/uscms_data/d3/escheull/ae_outputs/",

  "ae_latent": 16,
  "enc_nodes": [512, 256],
  "dec_nodes": [256, 512],
  "alpha": 0.0,
  "ae_lr": 1e-3,
  "epochs": 20,
  "batch_size": 4096,
  "seed": 123,
}


#train the ae
ae_out = run_ae(cfg_ae)

#print when done where you saved everything
print("Saved AE files here: ")
print(os.path.join(cfg_ae["outdir"], "ae_scores_bkg_train.npy"))
print(os.path.join(cfg_ae["outdir"], "ae_scores_bkg_test.npy"))