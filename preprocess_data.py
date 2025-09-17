import json
import uproot
import numpy as np
import h5py as h5
import argparse
import awkward as ak

#arguments for python file (to run in condor in a loop)
parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
parser.add_argument("--dataset", default="2024I")
parser.add_argument("--out", required=True)
args = parser.parse_args()

#top n for each event
top_objects = {
    "Electrons": 4,
    "Muons": 4,
    "Photons": 4,
    "Jets": 10,
    "FatJets": 10,
    "MET": 1
}

#branches saved in h5 (scouting)
scout_branches = {
    "Electrons": ["ScoutingElectron_pt", "ScoutingElectron_eta", "ScoutingElectron_phi", "event"],
    "MuonsVtx": ["ScoutingMuonVtx_pt", "ScoutingMuonVtx_eta", "ScoutingMuonVtx_phi", "event"],
    "MuonsNoVtx": ["ScoutingMuonNoVtx_pt", "ScoutingMuonNoVtx_eta", "ScoutingMuonNoVtx_phi", "event"],
    "Photons": ["ScoutingPhoton_pt", "ScoutingPhoton_eta", "ScoutingPhoton_phi", "event"],
    "Jets": ["ScoutingPFJetRecluster_pt", "ScoutingPFJetRecluster_eta", "ScoutingPFJetRecluster_phi", "event"],
    "FatJets": ["ScoutingFatPFJetRecluster_pt", "ScoutingFatPFJetRecluster_eta", "ScoutingFatPFJetRecluster_phi", "event"],
    "MET": ["ScoutingMET_pt", "ScoutingMET_phi"]
}

#load the json files
with open(args.json) as f:
    filelist = json.load(f)
files = list(filelist[args.dataset]["files"].keys())

#init list
data = {}

#function to pick top N objects per event
def sort_top_n(name, N):
    #per event jagged arrays (using awkward arrays)
    pts  = arrays[f"{name}_pt"]
    etas = arrays[f"{name}_eta"]
    phis = arrays[f"{name}_phi"]
    events = arrays["event"]

    result = {}

    #loop over events
    for i in range(len(events)):
        event = int(events[i])

        #extract each event objects lists as np arrays
        pt_i = np.asarray(pts[i], dtype=np.float32)
        eta_i = np.asarray(etas[i], dtype=np.float32)
        phi_i = np.asarray(phis[i], dtype=np.float32)

        #if there are objects, pick top N by pt
        if pt_i.size:
            #sort to descending and take first N
            pt_order = np.argsort(pt_i)[-N:][::-1]  
            objects = [(float(pt_i[j]), float(eta_i[j]), float(phi_i[j])) for j in pt_order]
        else:
            objects = []

        #pad entries as 0 if there are less than N objecst in a given event
        while len(objects) < N:
            objects.append((0.0, 0.0, 0.0))

        result[event] = objects
    return result

#loop thru files
for fname in files:
    try:
        #open a root file and access the events tree
        with uproot.open(fname)["Events"] as tree:
            #load branches into np arrays
            arrays = tree.arrays(
                expressions=sum(scout_branches.values(), []),
                library="ak"
            )
            print("python still running")
            #events
            events = arrays["event"]

            #combine muons with and without a vertex
            muon_vtx = sort_top_n("ScoutingMuonVtx", top_objects["Muons"])
            muon_no_vtx = sort_top_n("ScoutingMuonNoVtx", top_objects["Muons"])

            #init data list
            muon_data = {}

            for event in set(muon_vtx.keys()).union(muon_no_vtx.keys()):
                muons = muon_vtx.get(event, []) + muon_no_vtx.get(event, [])
                #take top N for the muons combined
                muons = sorted(muons, key=lambda x: -x[0])[:top_objects["Muons"]]
                while len(muons) < top_objects["Muons"]:
                    muons.append((0.0, 0.0, 0.0))
                muon_data[event] = muons

            #sort data for all other objects
            electron_data = sort_top_n("ScoutingElectron", top_objects["Electrons"])
            photon_data = sort_top_n("ScoutingPhoton", top_objects["Photons"])
            jet_data = sort_top_n("ScoutingPFJetRecluster", top_objects["Jets"])
            fat_jet_data = sort_top_n("ScoutingFatPFJetRecluster", top_objects["FatJets"])

            #met only has pt and phi
            met_pt  = arrays["ScoutingMET_pt"]
            met_phi = arrays["ScoutingMET_phi"]

            #events in data lits
            for i, _ in enumerate(events):
                key = len(data)   
                data[key] = []
                data[key].extend(electron_data.get(int(events[i]),[(0.0,0.0,0.0)]*4))
                data[key].extend(muon_data.get(int(events[i]), [(0.0,0.0,0.0)]*4))
                data[key].extend(photon_data.get(int(events[i]), [(0.0,0.0,0.0)]*4))
                data[key].extend(jet_data.get(int(events[i]), [(0.0,0.0,0.0)]*10))
                data[key].extend(fat_jet_data.get(int(events[i]), [(0.0,0.0,0.0)]*10))
                data[key].append((float(met_pt[i]), 0.0, float(met_phi[i])))

    except Exception as e:
        print(f"Error with {fname}: {e}")

print("Converting to h5")

#convert to np array for each event
DATA = np.array([data[event] for event in data], dtype=np.float32)

#check shape (should be (n_events, 33, 3)
print("Final shape:", DATA.shape)

n_events = DATA.shape[0]
#returns random permutation of indices 
rng = np.random.default_rng(1337)
idx = rng.permutation(n_events)

#take 80 percent data for training
n_train = int(0.8 * n_events)
idx_train, idx_test = idx[:n_train], idx[n_train:]

#split data into training and testing
x_train = DATA[idx_train]
x_test = DATA[idx_test]

#Normalization from Diptarko (maybe ask about this)
#Ignores padded objects for statistics
#Robust bounds from train (compute 1st and 99th percentiles and map to -1, +1)
#Robust scaling  is done to make training stable (less saturation)
percentiles = (1.0, 99.0)
l1, h1 = -1.0, 1.0

n_slots = x_train.shape[1]
norm_scale = np.ones((n_slots, 3), dtype=np.float32)
norm_bias = np.zeros((n_slots, 3), dtype=np.float32)

#masks of padded rows (capture BEFORE transform)
pad_train = (x_train[..., 0] == 0.0)
pad_test = (x_test[...,  0] == 0.0)

for i in range(n_slots):
    # real (non-padded) rows in this slot on Train
    mask = ~pad_train[:, i]
    if np.any(mask):
        vals = x_train[:, i, :][mask]
        l0 = np.percentile(vals, percentiles[0], axis=0)
        h0 = np.percentile(vals, percentiles[1], axis=0)

        rng = h0 - l0
        rng[rng == 0] = 1.0  # avoid divide-by-zero

        denom = (h1 - l1)
        norm_scale[i] = rng / denom
        norm_bias[i]  = (l0 * h1 - h0 * l1) / denom
    else:
        # no real objects in this slot on Train → identity transform
        norm_scale[i] = 1.0
        norm_bias[i]  = 0.0

# apply normalization
x_train = (x_train - norm_bias) / norm_scale
x_test  = (x_test  - norm_bias) / norm_scale

# keep padded triplets exactly zero after normalization
x_train[pad_train] = 0.0
x_test[pad_test]   = 0.0


#check shape (should be (n_events, 32, 3)
print("Final shape:", DATA.shape)

#convert to h5 file
with h5.File(args.out, "w") as f:
    bkg_group = f.create_group("Background_data")

    test_group = bkg_group.create_group("Test")
    test_group.create_dataset("DATA", data=x_test, compression="gzip")

    train_group = bkg_group.create_group("Train")
    train_group.create_dataset("DATA", data=x_train, compression="gzip")

    norm_group = f.create_group("Normalization")
    norm_group.create_dataset("norm_bias", data=norm_bias, compression="gzip")
    norm_group.create_dataset("norm_scale", data=norm_scale, compression="gzip")

print(f"Saved to {args.out}")             
                
            



                    
                    



                






                

