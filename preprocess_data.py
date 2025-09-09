import json
import uproot
import numpy as np
import h5py as h5
import argparse

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

#loop thru files
for fname in files:
    try:
        #open a root file and access the events tree
        with uproot.open(fname)["Events"] as tree:
            #load branches into np arrays
            arrays = tree.arrays(
                expressions=sum(scout_branches.values(), []),
                library="np"
            )
            #events
            events = arrays["event"]

            #function to pick top N objects per event
            def top_n(name, n):
                pts = arrays[f"{name}_pt"]
                etas = arrays[f"{name}_eta"]
                pts = arrays[f"{name}_phi"]
                events = arrays["event"]

                per_event = {}

                #put objects into events
                for pt, eta, phi, event in zip(pts, etas, phis, events):
                    if event not in per_event:
                        per_event[event] = []
                        per_event[event].append((pt, eta, phi))



                






                

