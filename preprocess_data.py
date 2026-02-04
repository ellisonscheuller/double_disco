import json
import uproot
import numpy as np
import h5py as h5
import argparse
import awkward as ak

JET_RADIUS = 0.4
FATJET_RADIUS  = 0.8
ALLOW_OVERLAP = False

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True, help="Path to dataset manifest JSON")
parser.add_argument("--dataset", default="2024I", help="Dataset key in manifest")
parser.add_argument("--out", required=True, help="Output HDF5 path")
args = parser.parse_args()

top_objects = {
    "Electrons": 4,
    "Muons": 4,
    "Photons": 4,
    "Jets": 10,
    "FatJets": 10,
    "MET": 1
}

scout_branches = {
    "Electrons": ["ScoutingElectron_pt", "ScoutingElectron_eta", "ScoutingElectron_phi", "event"],
    "MuonsVtx":  ["ScoutingMuonVtx_pt", "ScoutingMuonVtx_eta", "ScoutingMuonVtx_phi", "event"],
    "MuonsNoVtx":["ScoutingMuonNoVtx_pt","ScoutingMuonNoVtx_eta","ScoutingMuonNoVtx_phi","event"],
    "Photons":   ["ScoutingPhoton_pt", "ScoutingPhoton_eta", "ScoutingPhoton_phi", "event"],
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
        "ScoutingFatPFJetRecluster_particleNet_prob_Hcc",
        "ScoutingFatPFJetRecluster_particleNet_prob_Hqq",
        "ScoutingFatPFJetRecluster_particleNet_prob_QCD",
        "ScoutingFatPFJetRecluster_mass",
        "ScoutingFatPFJetRecluster_nConstituents",
        "ScoutingFatPFJetRecluster_nElectrons",
        "ScoutingFatPFJetRecluster_nMuons",
        "ScoutingFatPFJetRecluster_nPhotons",
        "event"
    ],
    "MET": ["ScoutingMET_pt", "ScoutingMET_phi"]
}

#because some arrays have only pt, eta, phia nd some have more features
#this just basically detects how many extra per object features exist 
#and sets a fixed vector size so all objects can be stacked into a single tensor
extras_by_prefix = {}
for coll, branches in scout_branches.items():
    sample = next((b for b in branches if b != "event"), None)
    if sample is None or "_" not in sample:
        continue
    prefix = sample.rsplit("_", 1)[0]
    extras_by_prefix.setdefault(prefix, [])
    for b in branches:
        if b == "event" or not b.startswith(prefix + "_"):
            continue
        suffix = b[len(prefix) + 1:]
        if suffix not in ("pt", "eta", "phi"):
            extras_by_prefix[prefix].append(suffix)
for p in extras_by_prefix:
    extras_by_prefix[p] = list(dict.fromkeys(extras_by_prefix[p]))

N_FEATURES = 3 + (max((len(v) for v in extras_by_prefix.values()), default=0))

#printout so you can debug
print("==== Configuration Summary ====")
print(f"Dataset key: {args.dataset}")
print(f"Output file: {args.out}")
print(f"Top-N per collection: {top_objects}")
print(f"Jet radius: {JET_RADIUS}")
print(f"FatJet radius: {FATJET_RADIUS}")
print(f"Allow overlap: {ALLOW_OVERLAP}")
print(f"Detected extras by prefix:")
for p, ex in extras_by_prefix.items():
    print(f"  - {p}: {ex if ex else 'None'}")
print(f"N_FEATURES (per slot) : {N_FEATURES}")
print("================================\n")

#compute delta phi with periodic boundary conditions
def dphi(a, b):
    d = a - b
    return (d + np.pi) % (2*np.pi) - np.pi

#delta R = sqrt((delta eta)^2 + (delta_phi)^2)
def deltaR(eta1, phi1, eta2, phi2):
    return np.hypot(eta1 - eta2, dphi(phi1, phi2))

#lepton filtering
def clean_collection_vs_leptons(arrays, coll_prefix, radius):
    """
    Remove objects in coll_prefix within delta R<radius of ANY lepton (Electrons + Muons vtx/no-vtx).
    Returns a new awkward record array with that collection's branches filtered.
    """
    c_eta = arrays[f"{coll_prefix}_eta"]
    c_phi = arrays[f"{coll_prefix}_phi"]

    # Leptons 
    e_eta = arrays["ScoutingElectron_eta"] if "ScoutingElectron_eta" in arrays.fields else ak.Array([])
    e_phi = arrays["ScoutingElectron_phi"] if "ScoutingElectron_phi" in arrays.fields else ak.Array([])
    mv_eta = arrays["ScoutingMuonVtx_eta"] if "ScoutingMuonVtx_eta" in arrays.fields else ak.Array([])
    mv_phi = arrays["ScoutingMuonVtx_phi"] if "ScoutingMuonVtx_phi" in arrays.fields else ak.Array([])
    mn_eta = arrays["ScoutingMuonNoVtx_eta"] if "ScoutingMuonNoVtx_eta" in arrays.fields else ak.Array([])
    mn_phi = arrays["ScoutingMuonNoVtx_phi"] if "ScoutingMuonNoVtx_phi" in arrays.fields else ak.Array([])

    keep_masks = []
    n_events = len(arrays["event"])
    for i in range(n_events):
        eta_c = np.asarray(c_eta[i], dtype=np.float32)
        phi_c = np.asarray(c_phi[i], dtype=np.float32)

        lep_sets = []
        if len(e_eta) > 0: lep_sets.append((np.asarray(e_eta[i], dtype=np.float32), np.asarray(e_phi[i], dtype=np.float32)))
        if len(mv_eta) > 0: lep_sets.append((np.asarray(mv_eta[i], dtype=np.float32), np.asarray(mv_phi[i], dtype=np.float32)))
        if len(mn_eta) > 0: lep_sets.append((np.asarray(mn_eta[i], dtype=np.float32), np.asarray(mn_phi[i], dtype=np.float32)))

        if eta_c.size == 0:
            keep_masks.append(np.array([], dtype=bool))
            continue
        if not lep_sets:
            keep_masks.append(np.ones_like(eta_c, dtype=bool))
            continue

        min_dr = np.full_like(eta_c, np.inf, dtype=np.float32)
        for (eta_l, phi_l) in lep_sets:
            for k in range(len(eta_l)):
                dr = deltaR(eta_c, phi_c, eta_l[k], phi_l[k])
                min_dr = np.minimum(min_dr, dr.astype(np.float32))
        keep_masks.append(min_dr >= radius)

    # Replace all branches with this prefix 
    out = arrays
    for key in arrays.fields:
        if key.startswith(coll_prefix + "_"):
            arr = arrays[key]
            fil = []
            for i in range(len(arr)):
                ai = np.asarray(arr[i])
                ki = keep_masks[i]
                if ki.shape[0] == ai.shape[0]:
                    fil.append(ai[ki])
                else:
                    fil.append(ai)
            out = ak.with_field(out, ak.Array(fil), where=key)
    return out
def sort_top_n(name, N):
    pts  = arrays[f"{name}_pt"]
    etas = arrays[f"{name}_eta"] if f"{name}_eta" in arrays.fields else None
    phis = arrays[f"{name}_phi"] if f"{name}_phi" in arrays.fields else None
    events = arrays["event"]

    extra_suffixes = extras_by_prefix.get(name, [])
    extra_arrays = []
    for sfx in extra_suffixes:
        key = f"{name}_{sfx}"
        extra_arrays.append(arrays[key] if key in arrays.fields else ak.zeros_like(pts))

    result = {}
    for i in range(len(events)):
        event = int(events[i])
        pt_i = np.asarray(pts[i], dtype=np.float32)
        eta_i = np.asarray(etas[i], dtype=np.float32) if etas is not None else np.zeros_like(pt_i, dtype=np.float32)
        phi_i = np.asarray(phis[i], dtype=np.float32) if phis is not None else np.zeros_like(pt_i, dtype=np.float32)
        extras_ev = [np.asarray(ea[i], dtype=np.float32) for ea in extra_arrays]

        if pt_i.size:
            order = np.argsort(pt_i)[-N:][::-1]
            objects = []
            for j in order:
                row = [float(pt_i[j]), float(eta_i[j]), float(phi_i[j])]
                for ex in extras_ev:
                    row.append(float(ex[j]) if j < len(ex) else 0.0)
                if len(row) < N_FEATURES:
                    row += [0.0] * (N_FEATURES - len(row))
                else:
                    row = row[:N_FEATURES]
                objects.append(tuple(row))
        else:
            objects = []

        while len(objects) < N:
            objects.append(tuple([0.0] * N_FEATURES))

        result[event] = objects

    return result

with open(args.json) as f:
    filelist = json.load(f)
files = list(filelist[args.dataset]["files"].keys())
print(f"Found {len(files)} files to process.")

data = {}

for idx_file, fname in enumerate(files, 1):
    try:
        print(f"[{idx_file}/{len(files)}] Opening: {fname}")
        with uproot.open(fname)["Events"] as tree:
            exprs = sum(scout_branches.values(), [])
            arrays = tree.arrays(expressions=exprs, library="ak")
            n_ev = len(arrays["event"])
            print(f" - Loaded {n_ev} events")
            print("Branches actually read:", ", ".join(sorted(arrays.fields)))

            # Clean (or skip) using hard-coded flags
            if ALLOW_OVERLAP:
                arrays_for_jets    = arrays
                arrays_for_fatjets = arrays
                print("Overlap allowed (hard-coded): skipping jet/fatjet cleaning")
            else:
                arrays_for_jets = clean_collection_vs_leptons(
                    arrays, coll_prefix="ScoutingPFJetRecluster", radius=JET_RADIUS
                )
                arrays_for_fatjets = clean_collection_vs_leptons(
                    arrays, coll_prefix="ScoutingFatPFJetRecluster", radius=FATJET_RADIUS
                )
                print(f"  - Jet cleaning     : ΔR < {JET_RADIUS} to any lepton → removed")
                print(f"  - FatJet cleaning  : ΔR < {FATJET_RADIUS} to any lepton → removed")

            # Build Top-N (switch arrays appropriately for jets/fatjets)
            muon_vtx = sort_top_n("ScoutingMuonVtx", top_objects["Muons"])
            muon_no_vtx = sort_top_n("ScoutingMuonNoVtx", top_objects["Muons"])
            electron_data = sort_top_n("ScoutingElectron", top_objects["Electrons"])
            photon_data = sort_top_n("ScoutingPhoton", top_objects["Photons"])

            arrays_backup = arrays

            arrays = arrays_for_jets
            jet_data = sort_top_n("ScoutingPFJetRecluster", top_objects["Jets"])

            arrays = arrays_for_fatjets
            fat_jet_data = sort_top_n("ScoutingFatPFJetRecluster", top_objects["FatJets"])

            arrays = arrays_backup

            # Merge Muons (vtx + no-vtx) goes to Top 4
            muon_data = {}
            all_mu_ev = set(muon_vtx.keys()).union(muon_no_vtx.keys())
            for ev in all_mu_ev:
                muons = muon_vtx.get(ev, []) + muon_no_vtx.get(ev, [])
                muons = sorted(muons, key=lambda x: -x[0])[:top_objects["Muons"]]
                while len(muons) < top_objects["Muons"]:
                    muons.append(tuple([0.0] * N_FEATURES))
                muon_data[ev] = muons

            # MET
            met_pt = arrays["ScoutingMET_pt"]
            met_phi = arrays["ScoutingMET_phi"]

            # Stack: Electrons, Muons, Photons, Jets, FatJets, MET 
            events = arrays["event"]
            added = 0
            for i, _ in enumerate(events):
                key = len(data)
                ev  = int(events[i])
                data[key] = []
                data[key].extend(electron_data.get(ev, [(0.0,)*N_FEATURES]*4))
                data[key].extend(muon_data.get(ev, [(0.0,)*N_FEATURES]*4))
                data[key].extend(photon_data.get(ev, [(0.0,)*N_FEATURES]*4))
                data[key].extend(jet_data.get(ev, [(0.0,)*N_FEATURES]*10))
                data[key].extend(fat_jet_data.get(ev, [(0.0,)*N_FEATURES]*10))

                met_vals = [float(met_pt[i]), 0.0, float(met_phi[i])]
                if len(met_vals) < N_FEATURES:
                    met_vals += [0.0] * (N_FEATURES - len(met_vals))
                data[key].append(tuple(met_vals))
                added += 1

            print(f" - Appended {added} rows (events) from this file")

    except Exception as e:
        print(f"[WARN] Error with {fname}: {e}")                    

#make h5
print("Converting to h5 (assembling array) ...")
DATA = np.array([data[event] for event in data], dtype=np.float32)
print(f"Final DATA shape (n_events, 33, N_FEATURES): {DATA.shape}")
print(f"Total events aggregated: {DATA.shape[0]}")

# Same split as your original
rng = np.random.default_rng(1337)
idx = rng.permutation(DATA.shape[0])
n_train = int(0.8 * DATA.shape[0])
idx_train, idx_test = idx[:n_train], idx[n_train:]
print(f"Split: train={len(idx_train)}  test={len(idx_test)}")

x_train = DATA[idx_train]
x_test  = DATA[idx_test]

print(f"Writing HDF5 to: {args.out}")
with h5.File(args.out, "w") as f:
    bkg_group = f.create_group("Background_data")
    bkg_group.create_group("Test").create_dataset("DATA", data=x_test, compression="gzip")
    bkg_group.create_group("Train").create_dataset("DATA", data=x_train, compression="gzip")

    meta = {
        "top_objects": top_objects,
        "scout_branches": scout_branches,
        "extras_by_prefix": extras_by_prefix,
        "N_FEATURES": int(N_FEATURES),
        "jet_radius": float(JET_RADIUS),
        "fatjet_radius": float(FATJET_RADIUS),
        "allow_overlap": bool(ALLOW_OVERLAP),
        "note": "RAW values; NO normalization; jets and fatjets cleaned vs ALL leptons unless ALLOW_OVERLAP=True"
    }
    f.attrs["meta_json"] = json.dumps(meta)

print(f"Saved to {args.out}")
print("Done.")                    



                






                

