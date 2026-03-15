import pandas as pd
import glob
import os
import json

svgs = glob.glob('assets/circuits/*.svg')
svg_bases = [os.path.basename(s).split('-')[0] for s in svgs]
svg_files = {os.path.basename(s): s for s in svgs}

circuits = pd.read_csv('data/raw/circuits.csv')
mapping = {}

# manually defined hard-to-match aliases
aliases = {
    'albert_park': 'melbourne',
    'villeneuve': 'montreal',
    'red_bull_ring': 'spielberg',
    'rodriguez': 'mexico-city',
    'spa': 'spa-francorchamps',
    'americas': 'austin',
    'marina_bay': 'marina-bay',
    'yas_marina': 'yas-marina',
    'miami': 'miami',
    'baku': 'baku',
    'jeddah': 'jeddah',
    'losail': 'lusail',
    'vegas': 'las-vegas'
}

for _, row in circuits.iterrows():
    c_ref = str(row['circuitRef'])
    loc = str(row['location']).lower().replace(' ', '-')
    
    target = aliases.get(c_ref, c_ref)
    
    # get all svgs for this target
    matches = [s for s in svg_files.keys() if s.startswith(target)]
    if not matches:
        matches = [s for s in svg_files.keys() if s.startswith(loc)]
        
    if matches:
        # take the highest number (most recent layout)
        try:
            best_match = sorted(matches, key=lambda x: int(x.split('-')[1].split('.')[0]))[-1]
            mapping[int(row['circuitId'])] = 'assets/circuits/' + best_match
        except (ValueError, IndexError):
            best_match = matches[-1]
            mapping[int(row['circuitId'])] = 'assets/circuits/' + best_match

print(f"Mapped {len(mapping)} circuits.")
with open('assets/circuit_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=4)
