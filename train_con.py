import torch
import torch.nn as nn
import datetime
import os
import logging
import argparse
import wandb
import numpy as np
from embedding.loss import InfoNCELoss
from embedding.models import TransformerEncoder, Projector
from embedding.preprocs import PFPreProcessor, PUPPIPreProcessor
from embedding.training import make_train_val_split, build_train_val_loaders, train_epoch, validate_epoch, EarlyStopping, cosine_schedule_with_warmup, NuRDCritic
from embedding.utils.data_utils import compute_normalization_constants
from embedding.utils.cfg_handler import train_config, data_config
from embedding.utils.data_utils import compute_class_weights, load_data

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JEPA") 
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/training_{timestamp}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def main(data_path: str, cfg: train_config, cfg_data: data_config):

    run = wandb.init(
        project="embedding_hlt", 
        config = {
            **cfg.get_entire_cfg(),
            **cfg_data.get_entire_cfg()
        },
    )
    is_sweep = cfg.is_sweep()
    if is_sweep:
        logger.info("Running in sweep mode with wandb.config overrides.")
    if not is_sweep:
        run.name = f"{cfg.get_model_name()}_{timestamp}"
        logger.info(f"Running in standard mode using config file values. Run name: {run.name}")

    num_epochs = cfg.hp("num_epochs", 400 if not is_sweep else 50)
    patience = cfg.hp("early_stopping_patience", 100 if not is_sweep else 20)
    val_split = cfg.get_trdata_cfg("val_split", 0.1)
    pairwise = cfg.get_trdata_cfg("pairwise", False)
    class_weights_setting = cfg_data.get("class_weights", None)
    pfcands = cfg_data.get("pfcands", True)

    # Sweepable
    num_heads = cfg.hp("num_heads", 8)
    num_layers = cfg.hp("num_layers", 4)
    embed_size = cfg.hp("embed_size", 128)
    latent_dim = cfg.hp("latent_dim", 6)
    proj_dim = cfg.hp("proj_dim", 12)
    linear_dim = cfg.hp("linear_dim", None)
    infonce_temp = cfg.hp("InfoNCE_temp", 0.05)
    contrastive_weight = cfg.hp("contrastive_weight", 0.05)
    lr = cfg.hp("lr", 1e-3)
    batch_size = cfg.hp("batch_size", 256)

    #nurd hyperparams
    nurd_lambda = float(cfg.hp("nurd_lambda", 0.0))
    nurd_steps = int(cfg.hp("nurd_critic_steps", 1)) 
    nurd_hidden = int(cfg.hp("nurd_hidden", 128)) 

    # Load and split
    feature_block, label_block = load_data(data_path, map_location=device)

    ae_scores = None
    ae_path = cfg.get_trdata_cfg("ae_scores_path", None)  # or argparse
    if ae_path is not None:
        ae_scores = torch.from_numpy(np.load(ae_path)).float()
        assert ae_scores.shape[0] == feature_block.shape[0], "AE scores length must match events in .pt"
    
    X_tr, y_tr, X_val, y_val, idx_tr, idx_val = make_train_val_split(  
        feature_block, label_block, val_size=val_split
    )

    ae_tr = ae_val = None  
    if ae_scores is not None: 
        idx_tr_cpu = idx_tr.detach().cpu()  
        idx_val_cpu = idx_val.detach().cpu()
        ae_tr = ae_scores.index_select(0, idx_tr.cpu())  
        ae_val = ae_scores.index_select(0, idx_val.cpu())  

    # Log num classes
    num_classes = int(label_block.max().item()) + 1
    class_count_tr = torch.bincount(y_tr, minlength=num_classes)
    logger.info("Class counts in training set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_tr[i].item()} events")
    class_count_val = torch.bincount(y_val, minlength=num_classes)
    logger.info("Class counts in validation set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_val[i].item()} events")

    # Build loaders (use train stats for both)
    norm_constants = compute_normalization_constants(X_tr) if not pfcands else {}
    train_loader, val_loader = build_train_val_loaders(
        X_tr, y_tr, X_val, y_val, device=device, batch_size=batch_size, pfcands=pfcands, ae_scores_tr=ae_tr, ae_scores_val=ae_val
    )

    preproc = PFPreProcessor(norm_constants).to(device) if pfcands else PUPPIPreProcessor(norm_constants).to(device)
    encoder = TransformerEncoder(
        preproc.num_features,
        embed_size, 
        latent_dim, 
        num_heads=num_heads,
        num_layers=num_layers,
        linear_dim=linear_dim, 
        num_tokens=feature_block.size(1) if linear_dim is not None else None,
        pairwise=pairwise,
        pre_processor=preproc
    ).to(device).train()
    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim*4)).to(device).train()
    classifier = nn.Linear(proj_dim, num_classes).to(device).train()

    class_weights = compute_class_weights(label_block, setting=class_weights_setting).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    criterion = InfoNCELoss(temperature=infonce_temp)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + 
        list(projector.parameters()) + 
        list(classifier.parameters()),
        lr=lr
    )

    # NuRD critic + optimizer only if enabled and AE scores exist
    nurd_critic = None
    nurd_critic_optimizer = None
    if (ae_scores is not None) and (nurd_lambda > 0.0):
        nurd_critic = NuRDCritic(z_dim=proj_dim, hidden=nurd_hidden).to(device)
        nurd_critic_optimizer = torch.optim.Adam(nurd_critic.parameters(), lr=lr)
        logger.info(f"NuRD enabled: lambda={nurd_lambda}, steps={nurd_steps}, hidden={nurd_hidden}")
    else:
        logger.info("NuRD disabled (either no ae_scores_path or nurd_lambda=0).")

    # Scheduler based on TRAIN steps
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val = float("inf")
    es = EarlyStopping(patience=patience, mode="min", min_delta=0.0)

    model_path = os.path.join(os.getcwd(), "checkpoints", f"{cfg.get_model_name()}_encoder_{timestamp}.pth")

    for epoch in range(num_epochs):
        tr = train_epoch(
            encoder, projector, classifier,
            ce_loss_fn, criterion,
            train_loader, norm_constants, device,
            optimizer, scheduler, contrastive_weight=contrastive_weight,
            pairwise=pairwise,
            nurd_critic=nurd_critic,
            nurd_critic_optimizer=nurd_critic_optimizer,
            nurd_lambda=nurd_lambda,
            nurd_critic_steps=nurd_steps,
        )
        va = validate_epoch(
            encoder, projector, classifier,
            ce_loss_fn, criterion,
            val_loader, norm_constants, device,
            contrastive_weight=contrastive_weight,
            pairwise=pairwise,
            nurd_critic=nurd_critic,
            nurd_lambda=nurd_lambda,
        )

        info_tr = tr.get("info", 0.0) 
        info_va = va.get("info", 0.0)  

        log_str = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: loss {tr['loss']:.6f}, InfoNCE {tr['nce']:.6f}, CrossEntropy {tr['ce']:.6f}, acc {tr['acc']:.4f} | "
            f"Val:   loss {va['loss']:.6f}, InfoNCE {va['nce']:.6f}, CrossEntropy {va['ce']:.6f}, acc {va['acc']:.4f}"
        )
        logger.info(log_str)

        # save best on validation loss 
        if va["loss"] < best_val:
            best_val = va["loss"]
            payload = {
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "norm_constants": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm_constants.items()},
            }

            
            if nurd_critic is not None:
                payload["nurd_critic"] = nurd_critic.state_dict()
                payload["nurd_critic_optimizer"] = nurd_critic_optimizer.state_dict()

            torch.save(payload, model_path)
            logger.info(f"Saved best encoder to: {model_path}")
        
        run.log(
            {
                "Train Loss": tr["loss"],
                "Train InfoNCE": tr["nce"],
                "Train CrossEntropy": tr["ce"],
                "Train Info (NuRD)": info_tr,
                "Train Accuracy": tr["acc"],
                "Val Loss": va["loss"],
                "Val InfoNCE": va["nce"],
                "Val CrossEntropy": va["ce"],
                "Val Info (NuRD)": info_va,
                "Val Accuracy": va["acc"],
            }, step=epoch
        )

        if es.step(va["loss"]):
            logger.info("Early stopping triggered.")
            break
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True, help="Path to the data config .yaml file") 
    parser.add_argument("--train_cfg", required=True, help="Path to the training config .yaml file")
    parser.add_argument("--data", required=True, help="Path to the input .pt file")
    args = parser.parse_args()

    tr_cfg = train_config(args.train_cfg)
    data_cfg = data_config(args.data_cfg)
    
    logger.info(f"Using train config file: {args.train_cfg}")
    logger.info(f"Entire train config: {tr_cfg.get_entire_cfg()}")
    
    logger.info(f"Using data processing config file: {args.data_cfg}")
    logger.info(f"Entire data processing config: {data_cfg.get_entire_cfg()}")

    main(args.data, tr_cfg, data_cfg)
