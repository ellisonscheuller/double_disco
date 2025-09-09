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
            top_objects = [(float(pt_i[j]), float(eta_i[j]), float(phi_i[j])) for j in pt_order]
        else:
            top_objects = []

        #pad entries as 0 if there are less than N objecst in a given event
        while len(top_objects) < N:
            objs.append((0.0, 0.0, 0.0))

        result[event] = objs
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
            for i, event in enumerate(events):
                if event not in data:
                    data[event] = []

                data[event].extend(electron_data.get(event, [(0.0., 0.0., 0.0.)]*4))
                data[event].extend(muon_data.get(event, [(0.0., 0.0., 0.0.)]*4))
                data[event].extend(photon_data.get(event, [(0.0., 0.0., 0.0.)]*4))
                data[event].extend(jet_data.get(event, [(0.0., 0.0., 0.0.)]*10))
                data[event].extend(fat_jet_data.get(event, [(0.0., 0.0., 0.0.)]*10))
                data[event].append((met_pt[i], 0.0., met_phi[i]))


    except Exception as e:
        print(f"Error with {fname}: {e}")

print("Converting to h5")

#convert to np array for each event
DATA = np.array([data[event] for event in data], dtype=np.float32)

#check shape (should be (n_events, 32, 3)
print("Final shape:", DATA.shape)

n_events = DATA.shape[0]
#returns random permutation of indices 
idx = rng.permutation(n_events)

#take 80 percent data for training
n_train = int(0.8 * n_events)
idx_train, idx_test = idx[:n_train], idx[n_train:]

#split data into training and testing
x_train = DATA[idx_train]
x_test = DATA[idx_test]

#TODO: normalization can go here but ask about it first

#check shape (should be (n_events, 32, 3)
print("Final shape:", DATA.shape)

#convert to h5 file
with h5py.File(args.out, "w") as f:
    bkg_group = f.create_group("Background_data")

    test_group = bkg_group.create_group("Test")
    test_group.create_dataset("DATA", data=x_test, compression="gzip")

    train_group = bkg_group.create_group("Train")
    train_group.create_dataset("DATA", data=x_train, compression="gzip")

    # norm_group = f.create_group("Normalization")
    # norm_group.create_dataset("norm_bias", data=bias, compression="gzip")
    # norm_group.create_dataset("norm_scale", data=scale, compression="gzip")

print(f"Saved to {args.out}")             
                
            



                    
                    



                






                

