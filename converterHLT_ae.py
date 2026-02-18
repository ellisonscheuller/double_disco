import argparse
import datetime
import logging
from pathlib import Path
import os
from tqdm import tqdm
import glob
import torch
import numpy as np
import uproot
import awkward as ak
from typing import Union

from embedding.utils.cfg_handler import data_config, join_remote
from embedding.utils.data_utils import softkill, EPS

from typing import List

def expand_input_to_file_paths(file_name: str, sample_dir: Path) -> List[str]:
    """
    Returns a list of *strings*.
    - If file_name is a .txt list, read it line-by-line.
      * If a line starts with root:// keep it as-is.
      * Otherwise treat it as a local/relative path under sample_dir.
    - Otherwise treat file_name as a local path/glob under sample_dir.
    """
    file_name = str(file_name).strip()

    def norm_one(s: str) -> str:
        s = s.strip()
        if not s:
            return ""
        if s.startswith("root://"):
            return s
        return os.fspath(sample_dir / s)

    # .txt list case
    if file_name.endswith(".txt"):
        txt_path = sample_dir / file_name
        if not txt_path.exists():
            raise FileNotFoundError(f"Filelist {txt_path} not found.")
        out = []
        with open(txt_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                out.append(norm_one(ln))
        return out

    # normal glob case
    p = sample_dir / file_name
    return [os.fspath(x) for x in glob.glob(os.fspath(p))]

uproot.source.xrootd.XRootDSource.timeout = 480

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("converterHLT")


# Event IDs for alignment
def gather_event_ids(tree: uproot.TTree, max_events: int = -1) -> torch.Tensor:
    """
    Return (N, 3) int64 tensor: [run, lumi, event]
    """
    arr = tree.arrays(["run", "luminosityBlock", "event"], entry_stop=max_events)
    run  = np.asarray(arr["run"], dtype=np.int64)
    lumi = np.asarray(arr["luminosityBlock"], dtype=np.int64)
    evt  = np.asarray(arr["event"], dtype=np.int64)
    out = np.stack([run, lumi, evt], axis=1)
    return torch.from_numpy(out)


# Original "object-level" processor from Roy's code (pt,eta,phi,dxy,btag,has_dxy,has_btag,label)
def process_particles(
    pt: ak.Array, eta: ak.Array, phi: ak.Array, dxy: ak.Array, btag: ak.Array,
    has_dxy: ak.Array, has_btag: ak.Array,
    label: int,
    n_objects: int = 200,
    sort_by_pt: bool = False,
) -> torch.Tensor:
    particles = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "dxy": dxy, "btag": btag, "has_dxy": has_dxy, "has_btag": has_btag}
    )
    if sort_by_pt:
        logger.info("Sorting particles by pt")
        particles = particles[ak.argsort(particles.pt, axis=1, ascending=False)]

    counts = ak.num(particles.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(
            f"{int(ak.sum(clipped_events))} events clipped; {lost} particles lost. Increase n_objects to avoid this."
        )

    padded = ak.pad_none(particles, n_objects, axis=1, clip=True)
    array = np.stack(
        [
            ak.to_numpy(padded["pt"]),
            ak.to_numpy(padded["eta"]),
            ak.to_numpy(padded["phi"]),
            ak.to_numpy(padded["dxy"]),
            ak.to_numpy(padded["btag"]),
            ak.to_numpy(padded["has_dxy"]),
            ak.to_numpy(padded["has_btag"]),
        ],
        axis=-1,
    )
    tensor = torch.tensor(array, dtype=torch.float32)
    label_tensor = torch.full((tensor.shape[0], n_objects, 1), label, dtype=torch.float32)
    return torch.cat([tensor, label_tensor], dim=-1)


def gather_particles(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    """
    Return a single awkward array of "objects" including:
      jets, muons, electrons, photons, MET
    fields: pt, eta, phi, dxy, btag, has_dxy, has_btag
    """

    # Jets
    jet_pt = tree["ScoutingPFJetRecluster_pt"].array(entry_stop=max_events)
    jet_eta = tree["ScoutingPFJetRecluster_eta"].array(entry_stop=max_events)
    jet_phi = tree["ScoutingPFJetRecluster_phi"].array(entry_stop=max_events)
    jet_dxy = ak.zeros_like(jet_pt)
    jet_btag = (
        tree["ScoutingPFJetRecluster_particleNet_prob_bb"].array(entry_stop=max_events)
        + tree["ScoutingPFJetRecluster_particleNet_prob_b"].array(entry_stop=max_events)
    )
    jet_has_dxy = ak.zeros_like(jet_pt, dtype=np.int8)
    jet_has_btag = ak.ones_like(jet_pt, dtype=np.int8)
    jets = ak.zip(
        {"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "dxy": jet_dxy, "btag": jet_btag,
         "has_dxy": jet_has_dxy, "has_btag": jet_has_btag}
    )

    # Muons (Vtx)
    mu_pt = tree["ScoutingMuonVtx_pt"].array(entry_stop=max_events)
    mu_eta = tree["ScoutingMuonVtx_eta"].array(entry_stop=max_events)
    mu_phi = tree["ScoutingMuonVtx_phi"].array(entry_stop=max_events)
    mu_dxy = tree["ScoutingMuonVtx_trk_dxy"].array(entry_stop=max_events)
    mu_btag = ak.zeros_like(mu_pt)
    mu_has_dxy = ak.ones_like(mu_pt, dtype=np.int8)
    mu_has_btag = ak.zeros_like(mu_pt, dtype=np.int8)
    muons = ak.zip(
        {"pt": mu_pt, "eta": mu_eta, "phi": mu_phi, "dxy": mu_dxy, "btag": mu_btag,
         "has_dxy": mu_has_dxy, "has_btag": mu_has_btag}
    )

    # Electrons
    e_pt = tree["ScoutingElectron_pt"].array(entry_stop=max_events)
    e_eta = tree["ScoutingElectron_eta"].array(entry_stop=max_events)
    e_phi = tree["ScoutingElectron_phi"].array(entry_stop=max_events)
    e_dxy = ak.zeros_like(e_pt)
    e_btag = ak.zeros_like(e_pt)
    e_has_dxy = ak.zeros_like(e_pt, dtype=np.int8)
    e_has_btag = ak.zeros_like(e_pt, dtype=np.int8)
    electrons = ak.zip(
        {"pt": e_pt, "eta": e_eta, "phi": e_phi, "dxy": e_dxy, "btag": e_btag,
         "has_dxy": e_has_dxy, "has_btag": e_has_btag}
    )

    # Photons
    g_pt = tree["ScoutingPhoton_pt"].array(entry_stop=max_events)
    g_eta = tree["ScoutingPhoton_eta"].array(entry_stop=max_events)
    g_phi = tree["ScoutingPhoton_phi"].array(entry_stop=max_events)
    g_dxy = ak.zeros_like(g_pt)
    g_btag = ak.zeros_like(g_pt)
    g_has_dxy = ak.zeros_like(g_pt, dtype=np.int8)
    g_has_btag = ak.zeros_like(g_pt, dtype=np.int8)
    photons = ak.zip(
        {"pt": g_pt, "eta": g_eta, "phi": g_phi, "dxy": g_dxy, "btag": g_btag,
         "has_dxy": g_has_dxy, "has_btag": g_has_btag}
    )

    # MET
    met_pt = tree["ScoutingMET_pt"].array(entry_stop=max_events)
    met_phi = tree["ScoutingMET_phi"].array(entry_stop=max_events)
    met_eta = ak.zeros_like(met_pt)
    met_dxy = ak.zeros_like(met_pt)
    met_btag = ak.zeros_like(met_pt)
    met_has_dxy = ak.zeros_like(met_pt, dtype=np.int8)
    met_has_btag = ak.zeros_like(met_pt, dtype=np.int8)
    met = ak.zip(
        {"pt": met_pt, "eta": met_eta, "phi": met_phi, "dxy": met_dxy, "btag": met_btag,
         "has_dxy": met_has_dxy, "has_btag": met_has_btag}
    )
    met = met[:, np.newaxis]

    return ak.concatenate([jets, muons, electrons, photons, met], axis=1)

#do the 23 AXo like objects for autoencoder training
def gather_objects_for_ae(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    def _topk_pad(pt, eta, phi, type_id: int, k: int):
        """Sort by pt within object type, then pad/clip to exactly k."""
        # sort descending pt (per event)
        order = ak.argsort(pt, axis=1, ascending=False)
        pt_s = pt[order]
        eta_s = eta[order]
        phi_s = phi[order]

        # pad/clip to k
        pt_p = ak.pad_none(pt_s,  k, axis=1, clip=True)
        eta_p = ak.pad_none(eta_s, k, axis=1, clip=True)
        phi_p = ak.pad_none(phi_s, k, axis=1, clip=True)

        # type field matches padded shape
        typ = ak.ones_like(pt_p) * type_id

        return ak.zip({"pt": pt_p, "eta": eta_p, "phi": phi_p, "type": typ})

    # --- Jets (10) ---
    jet_pt = tree["ScoutingPFJetRecluster_pt"].array(entry_stop=max_events)
    jet_eta = tree["ScoutingPFJetRecluster_eta"].array(entry_stop=max_events)
    jet_phi = tree["ScoutingPFJetRecluster_phi"].array(entry_stop=max_events)
    jets = _topk_pad(jet_pt, jet_eta, jet_phi, type_id=0, k=10)

    # --- Muons (4) ---
    mu_pt = tree["ScoutingMuonVtx_pt"].array(entry_stop=max_events)
    mu_eta = tree["ScoutingMuonVtx_eta"].array(entry_stop=max_events)
    mu_phi = tree["ScoutingMuonVtx_phi"].array(entry_stop=max_events)
    muons = _topk_pad(mu_pt, mu_eta, mu_phi, type_id=1, k=4)

    # --- Electrons (4) ---
    e_pt = tree["ScoutingElectron_pt"].array(entry_stop=max_events)
    e_eta = tree["ScoutingElectron_eta"].array(entry_stop=max_events)
    e_phi = tree["ScoutingElectron_phi"].array(entry_stop=max_events)
    electrons = _topk_pad(e_pt, e_eta, e_phi, type_id=2, k=4)

    # --- Photons (4) ---
    g_pt = tree["ScoutingPhoton_pt"].array(entry_stop=max_events)
    g_eta = tree["ScoutingPhoton_eta"].array(entry_stop=max_events)
    g_phi = tree["ScoutingPhoton_phi"].array(entry_stop=max_events)
    photons = _topk_pad(g_pt, g_eta, g_phi, type_id=3, k=4)

    # --- MET (1) ---
    met_pt  = tree["ScoutingMET_pt"].array(entry_stop=max_events)
    met_phi = tree["ScoutingMET_phi"].array(entry_stop=max_events)
    met_eta = ak.zeros_like(met_pt)

    # make them (N, 1)
    met_pt  = met_pt[:, np.newaxis]
    met_eta = met_eta[:, np.newaxis]
    met_phi = met_phi[:, np.newaxis]

    met = ak.zip({
        "pt": met_pt,
        "eta": met_eta,
        "phi": met_phi,
        "type": ak.ones_like(met_pt) * 4,
    })

    # Concatenate in a fixed, semantic order:
    # [10 jets][4 mu][4 e][4 gamma][1 MET] => (N, 23)
    return ak.concatenate([jets, muons, electrons, photons, met], axis=1)

def process_objects_for_ae(
    combined: ak.Array,
    label: int,
    n_objects: int = 200,
    sort_by_pt: bool = True,
) -> torch.Tensor:
    if sort_by_pt:
        combined = combined[ak.argsort(combined.pt, axis=1, ascending=False)]

    counts = ak.num(combined.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(
            f"{int(ak.sum(clipped_events))} events clipped; {lost} objects lost (AE view). Increase objects_n to avoid this."
        )

    padded = ak.pad_none(combined, n_objects, axis=1, clip=True)
    array = np.stack(
        [
            ak.to_numpy(padded["pt"]),
            ak.to_numpy(padded["eta"]),
            ak.to_numpy(padded["phi"]),
            ak.to_numpy(padded["type"]),
        ],
        axis=-1,
    )
    tensor = torch.tensor(array, dtype=torch.float32)
    label_tensor = torch.full((tensor.shape[0], n_objects, 1), label, dtype=torch.float32)
    return torch.cat([tensor, label_tensor], dim=-1)  # (N, nobj, 5)


# PF candidates
def construct_pf_features(branches: ak.Array) -> ak.Array:
    branches["ScoutingParticle_is_pf"] = ak.ones_like(branches["ScoutingParticle_pt"])

    branches["ScoutingMuonNoVtx_dxy"] = branches["ScoutingMuonNoVtx_trk_dxy"]
    branches["ScoutingMuonNoVtx_dxysig"] = branches["ScoutingMuonNoVtx_trk_dxy"] / (branches["ScoutingMuonNoVtx_trk_dxyError"] + EPS)
    branches["ScoutingMuonNoVtx_pdgId"] = branches["ScoutingMuonNoVtx_charge"] * 13
    branches["ScoutingMuonNoVtx_is_pf"] = ak.zeros_like(branches["ScoutingMuonNoVtx_pt"])

    branches["ScoutingElectron_dxy"] = -branches["ScoutingElectron_bestTrack_d0"]
    branches["ScoutingElectron_dxysig"] = ak.zeros_like(branches["ScoutingElectron_pt"])
    branches["ScoutingElectron_pdgId"] = branches["ScoutingElectron_bestTrack_charge"] * 11
    branches["ScoutingElectron_is_pf"] = ak.zeros_like(branches["ScoutingElectron_pt"])

    branches["ScoutingPhoton_dxy"] = ak.zeros_like(branches["ScoutingPhoton_pt"])
    branches["ScoutingPhoton_dxysig"] = ak.zeros_like(branches["ScoutingPhoton_pt"])
    branches["ScoutingPhoton_pdgId"] = ak.ones_like(branches["ScoutingPhoton_pt"]) * 22
    branches["ScoutingPhoton_is_pf"] = ak.zeros_like(branches["ScoutingPhoton_pt"])

    return branches


def gather_pfcands(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    branches = tree.arrays([
        "ScoutingParticle_pt",
        "ScoutingParticle_eta",
        "ScoutingParticle_phi",
        "ScoutingParticle_dxy",
        "ScoutingParticle_dxysig",
        "ScoutingParticle_pdgId",
        "ScoutingMuonNoVtx_pt",
        "ScoutingMuonNoVtx_eta",
        "ScoutingMuonNoVtx_phi",
        "ScoutingMuonNoVtx_trk_dxy",
        "ScoutingMuonNoVtx_trk_dxyError",
        "ScoutingMuonNoVtx_charge",
        "ScoutingElectron_pt",
        "ScoutingElectron_eta",
        "ScoutingElectron_phi",
        "ScoutingElectron_bestTrack_d0",
        "ScoutingElectron_bestTrack_charge",
        "ScoutingPhoton_pt",
        "ScoutingPhoton_eta",
        "ScoutingPhoton_phi",
    ], entry_stop=max_events)

    branches = construct_pf_features(branches)

    branch_prefixes = [
        "ScoutingParticle_",
        "ScoutingMuonNoVtx_",
        "ScoutingElectron_",
        "ScoutingPhoton_",
    ]
    feature_names = [
        "pt",
        "eta",
        "phi",
        "dxy",
        "dxysig",
        "is_pf",
        "pdgId",
    ]

    combined = ak.concatenate([
        ak.zip({ftr_name: branches[prefix + ftr_name] for ftr_name in feature_names})
        for prefix in branch_prefixes
    ], axis=1)

    logger.info("PF candidate fields: " + ", ".join(combined.fields))
    return combined


def process_pfcands(
    combined: ak.Array,
    label: int,
    n_objects: int = 200,
    sk_cell_size: Union[float, None] = None,
    sort_by_pt: bool = False,
) -> torch.Tensor:

    if sort_by_pt:
        logger.info("Sorting PF candidates by pt")
        combined = combined[ak.argsort(combined.pt, axis=1, ascending=False)]

    counts = ak.num(combined.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(f"{int(ak.sum(clipped_events))} events clipped; {lost} particles lost. Increase n_objects to avoid this.")
    padded = ak.pad_none(combined, n_objects, axis=1, clip=True)

    array = np.stack([ak.to_numpy(padded[field]) for field in padded.fields], axis=-1)
    tensor = torch.tensor(array, dtype=torch.float32)
    label_tensor = torch.full((tensor.shape[0], n_objects, 1), label, dtype=torch.float32)

    if sk_cell_size is not None:
        tensor = softkill(tensor, cell_size=sk_cell_size)

    return torch.cat([tensor, label_tensor], dim=-1)


def main(cfg: data_config, overwrite: bool = False):
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    sample_dir = Path(cfg["sample_dir"]).expanduser()
    redir = cfg.get("redir", "")
    n_objects = cfg.get("n_objects", 500)       # PF tokens
    objects_n = cfg.get("objects_n", 200)       # AE tokens 
    nevents_per_class = cfg.get("nevents_per_class", -1)

    pfcands = cfg.get("pfcands", True)
    sk_cell_size = cfg.get("sk_spacing", None)
    sort_by_pt = cfg.get("sort_by_pt", True)
    store_by_class = cfg.get("store_by_class", False)
    split = cfg.get("split", None)

    also_save_objects = cfg.get("also_save_objects", False)

    logger.info(f"PFCands mode: {pfcands}")
    logger.info(f"Soft-kill cell size: {sk_cell_size}")
    logger.info(f"Also save objects: {also_save_objects}")

    if split and store_by_class:
        raise ValueError("Cannot use both split and store_by_class options at the same time.")

    pf_tensors = {}
    obj_tensors = {}
    id_tensors = {}

    file_label_tuples = cfg.get_file_label_map()

    for file_name, label in tqdm(file_label_tuples, desc="Processing files"):
        file_paths = expand_input_to_file_paths(file_name, sample_dir)
        n_events_left = nevents_per_class

        for src in file_paths:
            if redir and (not src.startswith("root://")):
                src = join_remote(redir, src)
        
            tree = uproot.open(src)["Events"]

            ids = gather_event_ids(tree, max_events=n_events_left) if also_save_objects else None

            # PF view
            if pfcands:
                pf_evt = process_pfcands(
                    gather_pfcands(tree, max_events=n_events_left),
                    label=label,
                    n_objects=n_objects,
                    sk_cell_size=cfg.get("sk_spacing", None),
                    sort_by_pt=sort_by_pt,
                )
            else:
                comb = gather_particles(tree, max_events=n_events_left)
                pf_evt = process_particles(
                    comb["pt"], comb["eta"], comb["phi"], comb["dxy"], comb["btag"], comb["has_dxy"], comb["has_btag"],
                    label=label,
                    n_objects=n_objects,
                    sort_by_pt=sort_by_pt,
                )

            pf_tensors[label] = pf_tensors.get(label, []) + [pf_evt]

            # AE object view
            if also_save_objects:
                obj_combined = gather_objects_for_ae(tree, max_events=n_events_left)
                obj_evt = process_objects_for_ae(obj_combined, label=label, n_objects=objects_n, sort_by_pt=False)

                obj_tensors[label] = obj_tensors.get(label, []) + [obj_evt]
                id_tensors[label]  = id_tensors.get(label, []) + [ids]

                if obj_evt.shape[0] != pf_evt.shape[0]:
                    raise RuntimeError(f"Event count mismatch label {label}: pf {pf_evt.shape[0]} vs obj {obj_evt.shape[0]}")
                if ids.shape[0] != pf_evt.shape[0]:
                    raise RuntimeError(f"EventID count mismatch label {label}: ids {ids.shape[0]} vs pf {pf_evt.shape[0]}")

            n_events_left -= pf_evt.shape[0]
            if n_events_left <= 0:
                break

    pf_class = {label: torch.cat(chunks, dim=0) for label, chunks in pf_tensors.items()}

    if also_save_objects:
        obj_class = {label: torch.cat(chunks, dim=0) for label, chunks in obj_tensors.items()}
        id_class  = {label: torch.cat(chunks, dim=0) for label, chunks in id_tensors.items()}

    # clip
    for label in pf_class:
        if nevents_per_class > 0:
            pf_class[label] = pf_class[label][:nevents_per_class]
            if also_save_objects:
                obj_class[label] = obj_class[label][:nevents_per_class]
                id_class[label]  = id_class[label][:nevents_per_class]

    total = sum(t.shape[0] for t in pf_class.values())
    logger.info("Class event counts:")
    for label in pf_class:
        logger.info(f"  Label {label}: {pf_class[label].shape[0]} events ({round(pf_class[label].shape[0] / total * 100, 2)}%)")

    pf_full = torch.cat([pf_class[label] for label in pf_class], dim=0)
    pf_full = torch.nan_to_num(pf_full, nan=0.0, posinf=0.0, neginf=0.0)

    if also_save_objects:
        obj_full = torch.cat([obj_class[label] for label in obj_class], dim=0)
        obj_full = torch.nan_to_num(obj_full, nan=0.0, posinf=0.0, neginf=0.0)
        id_full = torch.cat([id_class[label] for label in id_class], dim=0)

        perm = torch.randperm(pf_full.shape[0])
        pf_full = pf_full[perm]
        obj_full = obj_full[perm]
        id_full = id_full[perm]
    else:
        pf_full = pf_full[torch.randperm(pf_full.shape[0])]

    output_prefix = cfg.get_ds_name()
    if output_prefix == "":
        output_prefix = "embedding_hlt_ssl"
        logger.warning(f"Dataset name not found in config; using default {output_prefix}.")

    out_path = Path(cfg.get("out_path", "./")).expanduser()
    os.makedirs(out_path, exist_ok=True)

    if split is not None:
        split_idx = int(cfg.get("split", 0.8) * pf_full.shape[0])

        pf_train = out_path / f"{output_prefix}_contrastive_train.pt"
        pf_test = out_path / f"{output_prefix}_contrastive_test.pt"

        if (pf_train.exists() or pf_test.exists()) and not overwrite:
            raise FileExistsError(f"Output files exist in {os.fspath(out_path)}. Use --overwrite.")

        torch.save(pf_full[:split_idx], os.fspath(pf_train))
        torch.save(pf_full[split_idx:], os.fspath(pf_test))

        if also_save_objects:
            obj_train = out_path / f"{output_prefix}_ae_obj_train.pt"
            obj_test = out_path / f"{output_prefix}_ae_obj_test.pt"
            id_train = out_path / f"{output_prefix}_eventid_train.pt"
            id_test = out_path / f"{output_prefix}_eventid_test.pt"

            torch.save(obj_full[:split_idx], os.fspath(obj_train))
            torch.save(obj_full[split_idx:], os.fspath(obj_test))
            torch.save(id_full[:split_idx], os.fspath(id_train))
            torch.save(id_full[split_idx:], os.fspath(id_test))

            logger.info("Saved paired outputs:")
            logger.info(f"{pf_train}")
            logger.info(f"{pf_test}")
            logger.info(f"{obj_train}")
            logger.info(f"{obj_test}")
            logger.info(f"{id_train}")
            logger.info(f"{id_test}")


    elif store_by_class:
        label_name_map = cfg.get_label_name_map()
        for label, name in label_name_map.items():
            full_fname = out_path / f"{output_prefix}_{name}_testds.pt"
            if full_fname.exists() and not overwrite:
                raise FileExistsError(f"Output exists: {os.fspath(full_fname)}. Use --overwrite.")
            event_labels = pf_full[:, 0, -1].to(torch.int64)
            class_tensor = pf_full[event_labels == int(label)]
            torch.save(class_tensor, os.fspath(full_fname))
    else:
        full_fname = out_path / f"{output_prefix}.pt"
        if full_fname.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {os.fspath(full_fname)}. Use --overwrite.")
        torch.save(pf_full, os.fspath(full_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config .yaml file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    cfg = data_config(args.config)

    log_filename = f"logs/converterHLT_{cfg.get_ds_name()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting conversion with config: {args.config}")
    main(cfg, overwrite=args.overwrite)