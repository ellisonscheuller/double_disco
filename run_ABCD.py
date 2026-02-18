import os
import random
import gc
import argparse
import numpy as np
import h5py as h5
import torch
import torch.nn as nn
import wandb
#from models.autoencoder import Autoencoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from embedding.models import TransformerEncoder, Projector
from embedding.preprocs import PFPreProcessor
from embedding.utils.data_utils import load_data, compute_normalization_constants


def abcd_counts(loss_1, loss_2, percent_1, percent_2):
    thresh_1 = np.quantile(loss_1, percent_1)
    thresh_2 = np.quantile(loss_2, percent_2)
    A = int(((loss_1 > thresh_1) & (loss_2 > thresh_2)).sum())
    B = int(((loss_1 > thresh_1) & (loss_2 <= thresh_2)).sum())
    C = int(((loss_1 <= thresh_1) & (loss_2 > thresh_2)).sum())
    D = int(((loss_1 <= thresh_1) & (loss_2 <= thresh_2)).sum())
    return thresh_1, thresh_2, A, B, C, D

def nonclosure_A(A, B, C, D, eps=1e-8):
    A_hat = (B * C) / max(D, eps)
    if A_hat <= 0:
        return np.inf, A_hat
    return (A - A_hat) / A_hat, A_hat

def profile_plot(ax, x, y, nbins=30, logx=False, min_per_bin=20, label="mean ± SE"):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)

    x = x[m]
    y = y[m]

    # bin along x (linear or log space)
    if logx:
        xu = np.log10(x)
    else:
        xu = x

    # uniform bins over the chosen coordinate
    lo = float(xu.min())
    hi = float(xu.max())
    if lo == hi:
        hi = np.nextafter(hi, np.inf)

    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # stats per bin
    mean, _, _ = binned_statistic(xu, y, statistic="mean", bins=edges)
    std,  _, _ = binned_statistic(xu, y, statistic="std",  bins=edges)
    cnt,  _, _ = binned_statistic(xu, y, statistic="count", bins=edges)

    sem = std / np.sqrt(np.maximum(cnt, 1))

    # keep well populated bins
    good = cnt >= min_per_bin
    xc = centers[good]
    ym = mean[good]
    ye = sem[good]

    # convert x axis back from log if needed
    if logx:
        xplot = 10.0 ** xc
        ax.set_xscale("log")
    else:
        xplot = xc

    ax.errorbar(xplot, ym, yerr=ye, fmt="o", ms=3, lw=1, capsize=2, label=label)
    ax.grid(alpha=0.3)
    return {"x": xplot, "mean": ym, "sem": ye, "count": cnt[good]}

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

#compute nurd-md scores for axis 2
def compute_md_scores(ckpt_path, data_pt_path, device="cpu", batch_size=4096):
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)

    # -----------------------------
    # Infer encoder hyperparams from checkpoint (so load_state_dict works)
    # -----------------------------
    enc_sd = ckpt["encoder"]

    # embed_size from cls_token [1,1,E] (fallback: input_proj out features)
    if "cls_token" in enc_sd:
        embed_size = int(enc_sd["cls_token"].shape[-1])
    else:
        embed_size = int(enc_sd["input_proj.weight"].shape[0])

    # latent_dim from bottleneck [latent_dim, embed_size]
    latent_dim = int(enc_sd["bottleneck.weight"].shape[0])

    # num_layers from max "layers.{i}." present
    layer_ids = []
    for k in enc_sd.keys():
        if k.startswith("layers."):
            try:
                layer_ids.append(int(k.split(".")[1]))
            except Exception:
                pass
    num_layers = (max(layer_ids) + 1) if len(layer_ids) else 0

    # num_heads from bias_mlp last layer bias size (your ckpt shows 4)
    # find one key like "layers.0.self_attn.bias_mlp.2.bias"
    num_heads = None
    for k, v in enc_sd.items():
        if k.endswith("self_attn.bias_mlp.2.bias") and hasattr(v, "shape"):
            num_heads = int(v.shape[0])
            break
    if num_heads is None:
        # fallback: assume 4 if not found
        num_heads = 4

    # pairwise flag: checkpoint has e_proj/f_proj keys when pairwise attention is used
    pairwise = any(("self_attn.e_proj" in k) or ("self_attn.f_proj" in k) for k in enc_sd.keys())

    # -----------------------------
    # Infer projector dims robustly (avoid fc1.weight assumptions)
    # -----------------------------
    proj_sd = ckpt["projector"]
    linear_w = [(k, v) for k, v in proj_sd.items() if hasattr(v, "ndim") and v.ndim == 2]
    if len(linear_w) == 0:
        raise KeyError(f"No 2D weight tensors found in projector state_dict keys: {list(proj_sd.keys())}")

    linear_w_sorted = sorted(linear_w, key=lambda kv: kv[0])
    first_w = linear_w_sorted[0][1]
    last_w  = linear_w_sorted[-1][1]
    proj_in_dim  = int(first_w.shape[1])
    proj_dim     = int(last_w.shape[0])

    # load data to infer input dims
    feature_block, label_block = load_data(data_pt_path, map_location=device)

    norm_constants = ckpt["norm_constants"]
    preproc = PFPreProcessor(norm_constants).to(device)

    encoder = TransformerEncoder(
        preproc.num_features,
        embed_size=embed_size,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        linear_dim=None,
        num_tokens=None,
        pairwise=pairwise,
        pre_processor=preproc
    ).to(device)

    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim*4)).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])

    encoder.eval()
    projector.eval()

    print(f"Loaded encoder config: embed_size={embed_size}, latent_dim={latent_dim}, num_heads={num_heads}, num_layers={num_layers}, pairwise={pairwise}")
    print("Computing embeddings...")

    embeddings = []

    with torch.no_grad():
        for i0 in range(0, len(feature_block), batch_size):
            xb = feature_block[i0:i0+batch_size].to(device)

            # NOTE: if this crashes next, we’ll add a proper mask (padding mask + CLS)
            latent = encoder(xb, None, None)
            z = projector(latent)
            z = F.normalize(z, dim=1)
            embeddings.append(z.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()

    #compute MD
    print("Computing Mahalanobis reference...")

    mu = embeddings.mean(axis=0)
    cov = np.cov(embeddings, rowvar=False)

    # regularize covariance
    eps = 1e-6
    cov += eps * np.eye(cov.shape[0])

    inv_cov = np.linalg.inv(cov)

    diffs = embeddings - mu
    md = np.einsum("bi,ij,bj->b", diffs, inv_cov, diffs)

    return md.astype(np.float32)



def ABCD(config):
    #go back into wandb (same project)
    print("Logging in to wandb...", flush=True)
    wandb.login(key="24d1d60ce26563c74d290d7b487cb104fc251271")
    wandb.init(project="AE vs. Contrastive ABCD",
               settings=wandb.Settings(_disable_stats=True),
               config=config)
    run_name = wandb.run.name
    print(f"Run name: {run_name}", flush=True)

    #output directorys
    outdir = config.get("outdir", "outputs_abcd")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    #load axis 1 arrays for sig and bkg
    ae_bkg = np.load(config["ae_scores_bkg_test"]).astype(np.float32).reshape(-1)
    ae_sig = None  # keep bkg-only for now

    #load axis 2 arrays for sig and bkg
    con_bkg = compute_md_scores(
        config["contrast_ckpt"],
        config["contrast_test_pt"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    con_sig = None  # keep bkg-only for now

    #check same length or it won't work (debugging)
    if len(con_bkg) != len(ae_bkg):
        raise ValueError(f"Contrastive and AE bkg arrays not the same length!")

    # NEW: only check signal lengths if you actually load them
    if (con_sig is not None) and (ae_sig is not None):
        if len(con_sig) != len(ae_sig):
            raise ValueError(f"Contrastive and AE sig arrays not the same length!")

    #masking (finite and positive)
    mask_bkg = np.isfinite(ae_bkg) & np.isfinite(con_bkg) & (ae_bkg > 0)

    # NEW: only make signal mask if signal arrays exist
    mask_sig = None
    if (ae_sig is not None) and (con_sig is not None):
        mask_sig = np.isfinite(ae_sig) & np.isfinite(con_sig) & (ae_sig > 0)

    #apply mask and rename to match axis for ABCD :)
    axis1_bkg, axis2_bkg = ae_bkg[mask_bkg], con_bkg[mask_bkg]

    # NEW: define signal axes only if present
    axis1_sig = axis2_sig = None
    if mask_sig is not None:
        axis1_sig, axis2_sig = ae_sig[mask_sig], con_sig[mask_sig]

    #ABCD vars
    percent = np.linspace(0.75, 0.98, 24)
    best = {"nonclosure": np.inf}
    min_A = int(config.get("min_A", 200))
    min_D = int(config.get("min_D", 1000))

    #do the 2D scan
    for p1 in percent:
        for p2 in percent:
            t1, t2, A, B, C, D = abcd_counts(axis1_bkg, axis2_bkg, p1, p2)
            if (A < min_A) or (D < min_D):
                continue
            nc, A_hat = nonclosure_A(A, B, C, D)
            if np.isfinite(nc) and abs(nc) < abs(best["nonclosure"]):
                best.update(dict(p1=p1, p2=p2, t1=t1, t2=t2, A=A, B=B, C=C, D=D,
                                 A_hat=A_hat, nonclosure=nc))

    # NEW: fail loudly if no working point found
    if "t1" not in best or "t2" not in best:
        raise RuntimeError("No ABCD working point found. Try lowering min_A/min_D or extending percent range.")

    #store optimized thresholds
    t1_opt = best["t1"]
    t2_opt = best["t2"]

    print(f"Optimized percents: p1={best['p1']:.3f}, p2={best['p2']:.3f}", flush=True)
    print(f"Optimized thresholds: t1={t1_opt:.6g}, t2={t2_opt:.6g}", flush=True)
    print(f"Nonclosure: {100.0*best['nonclosure']:.3f}%", flush=True)

    wandb.log({
        "ABCD/opt_p1": best["p1"],
        "ABCD/opt_p2": best["p2"],
        "ABCD/opt_t1": float(t1_opt),
        "ABCD/opt_t2": float(t2_opt),
        "ABCD/nonclosure": float(best["nonclosure"]),
        "ABCD/A": int(best["A"]),
        "ABCD/B": int(best["B"]),
        "ABCD/C": int(best["C"]),
        "ABCD/D": int(best["D"]),
    })

    #########
    #PLOTTING
    #########

    #2D histogram bkg only
    fig = plt.figure(figsize=(6,5))
    plt.hist2d(axis1_bkg, axis2_bkg, bins=200, norm=LogNorm(vmin=1), cmin=1)
    plt.axvline(t1_opt, color="black", ls="--")
    plt.axhline(t2_opt, color="black", ls="--")
    plt.xscale("log")
    plt.xlabel("AE loss")
    plt.ylabel("Contrastive score (MD)")
    plt.title("AE vs Contrastive (bkg only)")
    out = os.path.join(plot_dir, "hist2d_bkg.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Hists2D/bkg": wandb.Image(out)})

    # NEW: only do overlay plot if signal exists
    if axis1_sig is not None:
        fig, ax = plt.subplots(figsize=(6,5))
        hb = ax.hist2d(axis1_bkg, axis2_bkg, bins=200, norm=LogNorm(vmin=1), cmin=1)
        hs = ax.hist2d(axis1_sig, axis2_sig, bins=200, norm=LogNorm(vmin=1), cmin=1, alpha=0.65)
        ax.axvline(t1_opt, color="black", ls="--", lw=1.5)
        ax.axhline(t2_opt, color="black", ls="--", lw=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("AE loss")
        ax.set_ylabel("Contrastive score (MD)")
        ax.set_title("AE vs Contrastive")
        fig.colorbar(hb[3], ax=ax, pad=0.01, label="Background counts")
        fig.colorbar(hs[3], ax=ax, pad=0.08, label="Signal counts")
        out_overlay = os.path.join(plot_dir, "hist2d_overlay.png")
        fig.savefig(out_overlay, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"Hists2D/bkg_sig": wandb.Image(out_overlay)})

    #profile plots
    fig, ax = plt.subplots(figsize=(8, 6))
    profile_plot(ax, axis2_bkg, axis1_bkg, nbins=60, logx=False)
    ax.set_xlabel("Contrastive score (MD)")
    ax.set_ylabel("Mean AE loss")
    ax.set_yscale("log")
    ax.set_title("⟨AE loss⟩ vs contrastive")
    p1_path = os.path.join(plot_dir, "profile_AE_vs_contrastive.png")
    fig.savefig(p1_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/AE_vs_contrastive": wandb.Image(p1_path)})

    fig, ax = plt.subplots(figsize=(8, 6))
    profile_plot(ax, axis1_bkg, axis2_bkg, nbins=60, logx=True)
    ax.set_xlabel("AE loss")
    ax.set_ylabel("Mean contrastive score (MD)")
    ax.set_title("⟨contrastive⟩ vs AE loss")
    p2_path = os.path.join(plot_dir, "profile_contrastive_vs_AE.png")
    fig.savefig(p2_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/contrastive_vs_AE": wandb.Image(p2_path)})

    #1D scan just for plotting closure and S/sqrt(B)
    effs = []
    closure_ratio = []
    closure_unc = []
    s_over_sqrtb = []
    Ntot_bkg = float(len(axis1_bkg))

    for p in percent:
        t1, t2, A, B, C, D = abcd_counts(axis1_bkg, axis2_bkg, p, p)
        A_hat = (B*C)/max(D, 1e-8)
        ratio = A_hat/max(A, 1e-8)

        invA = 0.0 if A == 0 else 1.0/A
        invB = 0.0 if B == 0 else 1.0/B
        invC = 0.0 if C == 0 else 1.0/C
        invD = 0.0 if D == 0 else 1.0/D
        rel_var = invA + invB + invC + invD
        sigma = abs(ratio)*np.sqrt(rel_var) if rel_var > 0 else 0.0

        effs.append(A/max(Ntot_bkg, 1.0))
        closure_ratio.append(ratio)
        closure_unc.append(sigma)

        # NEW: bkg-only default for now
        if axis1_sig is None:
            A_sig = 0
        else:
            A_sig = int(((axis1_sig > t1) & (axis2_sig > t2)).sum())
        s_over_sqrtb.append(A_sig/np.sqrt(max(A, 1e-8)))

    effs = np.array(effs)
    closure_ratio = np.array(closure_ratio)
    closure_unc = np.array(closure_unc)
    s_over_sqrtb = np.array(s_over_sqrtb)

    #order everything for plotting
    order = np.argsort(effs)
    effs = effs[order]
    closure_ratio = closure_ratio[order]
    closure_unc = closure_unc[order]
    s_over_sqrtb = s_over_sqrtb[order]

    #compute optimized eff and S/sqrt(B) for putting on the plot
    eff_opt = best["A"]/max(Ntot_bkg, 1.0)
    ratio_opt = best["A_hat"]/max(best["A"], 1e-8)

    # bkg only 
    if axis1_sig is None:
        sigA_opt = 0
    else:
        sigA_opt = int(((axis1_sig > t1_opt) & (axis2_sig > t2_opt)).sum())
    s_over_sqrtb_opt = sigA_opt/np.sqrt(max(best["A"], 1e-8))

    #closure and s/sqrt(b) plots

    colors = ['g', 'b']
    fig_size = (8, 6)
    fs = 28
    fs_leg = 24

    fig, ax = plt.subplots(figsize=fig_size)

    # main curve
    ax.plot(effs, closure_ratio, c=colors[0], label="AE + Contrastive (MD)")

    # uncertainty band
    alpha_band = 0.5
    low = closure_ratio - closure_unc
    high = closure_ratio + closure_unc

    ax.fill_between(effs, low, high, facecolor=colors[0], alpha=alpha_band, interpolate=True)
    one = np.ones_like(effs)
    one_m = np.full_like(effs, 0.95)
    one_p = np.full_like(effs, 1.05)
    ax.plot(effs, one, linestyle='-',  color='black')
    ax.plot(effs, one_m, linestyle='--', color='black')
    ax.plot(effs, one_p, linestyle='--', color='black')
    ax.plot([eff_opt], [ratio_opt], marker='o', c='red', label='Optimized')
    ax.set_xlabel('Selection Efficiency (bkg A/Ntot)', fontsize=fs)
    ax.set_ylabel('Predicted Bkg. / True Bkg.', fontsize=fs)
    plt.ylim([0.0, 1.5])
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    plt.legend(loc="lower right", fontsize=fs_leg)
    closure_path = os.path.join(plot_dir, "cut_and_count_bkg_check.png")
    plt.savefig(closure_path, dpi=200, bbox_inches='tight')
    plt.close()
    wandb.log({"Closure/plot": wandb.Image(closure_path)})


    #S/sqrt(B) plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, s_over_sqrtb, color="red", label=r"$S/\sqrt{B}$")
    ax.plot([eff_opt], [s_over_sqrtb_opt], marker='o', color='black')
    ax.set_xlabel('Selection Efficiency (bkg A/Ntot)', fontsize=fs)
    ax.set_ylabel(r"$S/\sqrt{B}$", fontsize=fs)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    plt.legend(loc="best", fontsize=fs_leg)
    sig_path = os.path.join(plot_dir, "s_over_sqrtb_vs_bkg_eff.png")
    plt.savefig(sig_path, dpi=200, bbox_inches='tight')
    plt.close()
    wandb.log({"Signal/s_over_sqrtb_vs_bkg_eff": wandb.Image(sig_path)})

    wandb.finish()
    
if __name__ == "__main__":

    config = {
        "ae_scores_bkg_test": "/uscms_data/d3/escheull/ae_outputs/smcocktail_small/ae_scores_bkg_test.npy",
        "contrast_ckpt": "/uscms_data/d3/escheull/self-supervised-learning/checkpoints/embedding_hlt_smcocktail_small_nurd_encoder_20260218_122328.pth",
        "contrast_test_pt": "/uscms_data/d3/escheull/smcocktail_paired_small/out/embedding_hlt_smcocktail_small_paired_test.pt",
        "outdir": "outputs_abcd_small",
        "min_A": 200,
        "min_D": 1000,
    }

    ABCD(config)

