import json
import uproot
import numpy as np
import h5py as h5
import argparse
import awkward as ak

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
parser.add_argument("--dataset", default="2024I")
parser.add_argument("--out", required=True)
args = parser.parse_args()

#taking the top N of each object as such
top_objects = {
    "Electrons": 4,
    "Muons": 4,
    "Photons": 4,
    "Jets": 10,
    "FatJets": 10,
    "MET": 1
}

#branches chosen for the data file
scout_branches = {
    "Electrons": ["ScoutingElectron_pt", "ScoutingElectron_eta", "ScoutingElectron_phi", "event"],
    "MuonsVtx": ["ScoutingMuonVtx_pt", "ScoutingMuonVtx_eta", "ScoutingMuonVtx_phi", "event"],
    "MuonsNoVtx": ["ScoutingMuonNoVtx_pt", "ScoutingMuonNoVtx_eta", "ScoutingMuonNoVtx_phi", "event"],
    "Photons": ["ScoutingPhoton_pt", "ScoutingPhoton_eta", "ScoutingPhoton_phi", "event"],
    "Jets": [
        "ScoutingPFJetRecluster_pt",
        "ScoutingPFJetRecluster_eta",
        "ScoutingPFJetRecluster_phi",
        "ScoutingPFJetRecluster_particleNet_prob_b", 
        "ScoutingPFJetRecluster_particleNet_prob_bb",
        "ScoutingPFJetRecluster_particleNet_prob_c", 
        "ScoutingPFJetRecluster_particleNet_prob_cc", 
        "ScoutingPFJetRecluster_particleNet_prob_g", 
        "ScoutingPFJetRecluster_particleNet_prob_uds",
        "ScoutingPFJetRecluster_particleNet_prob_undef",
        "ScoutingPFJetRecluster_mass",
        "ScoutingPFJetRecluster_nConstituents",
        "ScoutingPFJetRecluster_nElectrons",
        "ScoutingPFJetRecluster_nMuons",
        "ScoutingPFJetRecluster_nPhotons",
        "event"
    ],
    "FatJets": [
        "ScoutingFatPFJetRecluster_pt", 
        "ScoutingFatPFJetRecluster_eta", 
        "ScoutingFatPFJetRecluster_phi", 
        "ScoutingFatPFJetRecluster_particleNet_prob_Hbb",
        "ScoutingFatPFJetRecluster_particleNet_prob_Hcc" , 
        "ScoutingFatPFJetRecluster_particleNet_prob_Hqq" , 
        "ScoutingFatPFJetRecluster_particleNet_prob_QCD" ,
        "ScoutingFatPFJetRecluster_mass",
        "ScoutingFatPFJetRecluster_nConstituents",
        "ScoutingFatPFJetRecluster_nElectrons",
        "ScoutingFatPFJetRecluster_nMuons",
        "ScoutingFatPFJetRecluster_nPhotons",
        "event"
               
    ],
    "MET": ["ScoutingMET_pt", "ScoutingMET_phi"]
}

#detect added branches to add to list
extras_by_prefix = {}
for coll, branches in scout_branches.items():
    # infer a ROOT prefix from the first non-"event" entry, e.g. "ScoutingPFJetRecluster"
    sample = next((b for b in branches if b != "event"), None)
    if sample is None or "_" not in sample:
        continue
    prefix = sample.rsplit("_", 1)[0]
    if prefix not in extras_by_prefix:
        extras_by_prefix[prefix] = []
    for b in branches:
        if b == "event" or not b.startswith(prefix + "_"):
            continue
        suffix = b[len(prefix) + 1:]
        if suffix not in ("pt", "eta", "phi"):
            extras_by_prefix[prefix].append(suffix)
# dedupe extras and summarize
for p in extras_by_prefix:
    extras_by_prefix[p] = list(dict.fromkeys(extras_by_prefix[p]))

# Global feature width for every slot (pt,eta,phi + extras)
N_FEATURES = 3 + (max((len(v) for v in extras_by_prefix.values()), default=0))

print("==== Configuration Summary ====")
print(f"Dataset key           : {args.dataset}")
print(f"Output file           : {args.out}")
print(f"Top-N per collection  : {top_objects}")
print(f"Detected extras by prefix:")
for p, ex in extras_by_prefix.items():
    print(f"  - {p}: {ex if ex else 'None'}")
print(f"N_FEATURES (per slot) : {N_FEATURES}")
print("================================\n")

#load root files
with open(args.json) as f:
    filelist = json.load(f)
files = list(filelist[args.dataset]["files"].keys())
print(f"Found {len(files)} files to process.")


data = {}

#pick top N by pt
def sort_top_n(name, N):
    # Core branch arrays (awkward)
    pts  = arrays[f"{name}_pt"]
    etas = arrays[f"{name}_eta"] if f"{name}_eta" in arrays.fields else None
    phis = arrays[f"{name}_phi"] if f"{name}_phi" in arrays.fields else None
    events = arrays["event"]

    # Get any declared extras (auto-discovered above)
    extra_suffixes = extras_by_prefix.get(name, [])
    extra_arrays = []
    for sfx in extra_suffixes:
        key = f"{name}_{sfx}"
        if key in arrays.fields:
            extra_arrays.append(arrays[key])
        else:
            # If a file lacks a declared branch, fill zeros (shape like pt)
            extra_arrays.append(ak.zeros_like(pts))

    result = {}

    # Loop over events (note: we keep prints outside this loop to avoid spam)
    for i in range(len(events)):
        event = int(events[i])

        # Convert awkward per-event lists to numpy arrays
        pt_i  = np.asarray(pts[i],  dtype=np.float32)
        eta_i = np.asarray(etas[i], dtype=np.float32) if etas is not None else np.zeros_like(pt_i, dtype=np.float32)
        phi_i = np.asarray(phis[i], dtype=np.float32) if phis is not None else np.zeros_like(pt_i, dtype=np.float32)
        extras_ev = [np.asarray(ea[i], dtype=np.float32) for ea in extra_arrays]

        # Build objects sorted by pt (desc), take top-N
        if pt_i.size:
            pt_order = np.argsort(pt_i)[-N:][::-1]
            objects = []
            for j in pt_order:
                row = [float(pt_i[j]), float(eta_i[j]), float(phi_i[j])]
                for ex in extras_ev:
                    row.append(float(ex[j]) if j < len(ex) else 0.0)
                # pad/truncate to N_FEATURES
                if len(row) < N_FEATURES:
                    row += [0.0] * (N_FEATURES - len(row))
                else:
                    row = row[:N_FEATURES]
                objects.append(tuple(row))
        else:
            objects = []

        # Pad to exactly N slots for this collection
        while len(objects) < N:
            objects.append(tuple([0.0] * N_FEATURES))

        result[event] = objects

    return result

#loop over files
for idx_file, fname in enumerate(files, 1):
    try:
        print(f"[{idx_file}/{len(files)}] Opening: {fname}")
        with uproot.open(fname)["Events"] as tree:
            # Build the flat list of branch names to read (your original structure)
            exprs = sum(scout_branches.values(), [])
            # Load all requested arrays in one go
            arrays = tree.arrays(expressions=exprs, library="ak")
            n_ev = len(arrays["event"])
            print(f"  - Loaded {n_ev} events")
            print("  - Branches actually read:",
                  ", ".join(sorted(arrays.fields)))

            # Build per-collection data (Top-N)
            muon_vtx = sort_top_n("ScoutingMuonVtx",         top_objects["Muons"])
            muon_no_vtx = sort_top_n("ScoutingMuonNoVtx",       top_objects["Muons"])
            electron_data= sort_top_n("ScoutingElectron",        top_objects["Electrons"])
            photon_data  = sort_top_n("ScoutingPhoton",          top_objects["Photons"])
            jet_data = sort_top_n("ScoutingPFJetRecluster",  top_objects["Jets"])
            fat_jet_data = sort_top_n("ScoutingFatPFJetRecluster", top_objects["FatJets"])

            # Combine muons with and without a vertex (your original logic)
            muon_data = {}
            all_mu_ev = set(muon_vtx.keys()).union(muon_no_vtx.keys())
            for ev in all_mu_ev:
                muons = muon_vtx.get(ev, []) + muon_no_vtx.get(ev, [])
                muons = sorted(muons, key=lambda x: -x[0])[:top_objects["Muons"]]
                while len(muons) < top_objects["Muons"]:
                    muons.append(tuple([0.0] * N_FEATURES))
                muon_data[ev] = muons

            # MET branches (scalar per event)
            met_pt = arrays["ScoutingMET_pt"]
            met_phi = arrays["ScoutingMET_phi"]

            # Stack the collections in a fixed order to one row per event
            events = arrays["event"]
            added = 0
            for i, _ in enumerate(events):
                key = len(data)   # new event index in output ordering
                ev  = int(events[i])
                data[key] = []
                data[key].extend(electron_data.get(ev, [(0.0,)*N_FEATURES]*4))
                data[key].extend(muon_data.get(ev,    [(0.0,)*N_FEATURES]*4))
                data[key].extend(photon_data.get(ev,  [(0.0,)*N_FEATURES]*4))
                data[key].extend(jet_data.get(ev,     [(0.0,)*N_FEATURES]*10))
                data[key].extend(fat_jet_data.get(ev, [(0.0,)*N_FEATURES]*10))

                # Pad MET row to N_FEATURES (pt, 0, phi, extras→0)
                met_vals = [float(met_pt[i]), 0.0, float(met_phi[i])]
                if len(met_vals) < N_FEATURES:
                    met_vals += [0.0] * (N_FEATURES - len(met_vals))
                data[key].append(tuple(met_vals))
                added += 1

            print(f"  - Appended {added} rows (events) from this file")

    except Exception as e:
        print(f"[WARN] Error with {fname}: {e}")

print("Converting to h5 (assembling array) ...")

#make data array
DATA = np.array([data[event] for event in data], dtype=np.float32)
print(f"Final DATA shape (n_events, 33, N_FEATURES): {DATA.shape}")

n_events = DATA.shape[0]
print(f"Total events aggregated: {n_events}")

#train/test split
rng = np.random.default_rng(1337)
idx = rng.permutation(n_events)
n_train = int(0.8 * n_events)
idx_train, idx_test = idx[:n_train], idx[n_train:]
print(f"Split: train={len(idx_train)}  test={len(idx_test)}")

x_train = DATA[idx_train]
x_test  = DATA[idx_test]

#robust scaling
percentiles = (1.0, 99.0)
l1, h1 = -1.0, 1.0

n_slots = x_train.shape[1]
n_feats = x_train.shape[2]  # dynamic feature count (3 + extras)
norm_scale = np.ones((n_slots, n_feats), dtype=np.float32)
norm_bias  = np.zeros((n_slots, n_feats), dtype=np.float32)

# masks of padded rows (capture BEFORE transform) — uses pt==0 invariant
pad_train = (x_train[..., 0] == 0.0)
pad_test  = (x_test[...,  0] == 0.0)

print("Computing per-slot normalization (train-only percentiles) ...")
num_identity = 0
for i in range(n_slots):
    mask = ~pad_train[:, i] 
    if np.any(mask):
        vals = x_train[:, i, :][mask]
        l0 = np.percentile(vals, percentiles[0], axis=0)
        h0 = np.percentile(vals, percentiles[1], axis=0)

        rngv = h0 - l0
        rngv[rngv == 0] = 1.0 

        denom = (h1 - l1)
        norm_scale[i] = rngv / denom
        norm_bias[i]  = (l0 * h1 - h0 * l1) / denom
    else:
        # no real objects in this slot on Train so identity transform
        num_identity += 1
        norm_scale[i] = 1.0
        norm_bias[i]  = 0.0

print(f"  - Slots with identity scaling (no real objects in train): {num_identity}/{n_slots}")

# Apply normalization
x_train = (x_train - norm_bias) / norm_scale
x_test  = (x_test  - norm_bias) / norm_scale

# Keep padded rows exactly zero after normalization
x_train[pad_train] = 0.0
x_test[pad_test]   = 0.0

print("Normalization applied.")
print(f"x_train shape: {x_train.shape}   x_test shape: {x_test.shape}")

#write h5
print(f"Writing HDF5 to: {args.out}")
with h5.File(args.out, "w") as f:
    bkg_group = f.create_group("Background_data")

    test_group = bkg_group.create_group("Test")
    test_group.create_dataset("DATA", data=x_test, compression="gzip")

    train_group = bkg_group.create_group("Train")
    train_group.create_dataset("DATA", data=x_train, compression="gzip")

    norm_group = f.create_group("Normalization")
    norm_group.create_dataset("norm_bias", data=norm_bias, compression="gzip")
    norm_group.create_dataset("norm_scale", data=norm_scale, compression="gzip")

    # Save a tiny metadata string so future-you remembers the layout decision
    meta = {
        "top_objects": top_objects,
        "scout_branches": scout_branches,  # what you actually asked uproot to read
        "extras_by_prefix": extras_by_prefix,
        "N_FEATURES": int(N_FEATURES)
    }
    f.attrs["meta_json"] = json.dumps(meta)

print(f"Saved to {args.out}")
print("Done.")
