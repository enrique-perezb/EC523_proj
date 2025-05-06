import os
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc

from brain_graph import BrainGraph

data_dir = '/projectnb/ec523/projects/proj_GS_LQ_EPB/data/T1w_segmented/HCP_processed/'
checkpoint_path = "hcp_brain_graph_dataset.pt"
completed_ids_path = "hcp_completed_subjects.txt"

## Load completed IDs
if os.path.exists(completed_ids_path):
    with open(completed_ids_path, 'r') as f:
        completed_subjects = set(line.strip() for line in f)
else:
    completed_subjects = set()

## Load participant ages
ages_path = '/projectnb/ec523/projects/proj_GS_LQ_EPB/data/T1w_segmented/HCP/ages.csv'
df = pd.read_csv(ages_path, dtype={'subject_id': str})
participant_ages = dict(zip(df['subject_id'], df['age']))

## Build list of participants to process
participants = []
for participant_id in os.listdir(data_dir):
    if participant_id in completed_subjects:
        continue

    base_path = os.path.join(data_dir, participant_id, participant_id, "mri")
    seg_file = os.path.join(base_path, "aparc.DKTatlas+aseg.deep.mgz")
    t1_file = os.path.join(base_path, "orig_nu.mgz")

    if not (os.path.exists(seg_file) and os.path.exists(t1_file)):
        continue

    age = participant_ages.get(participant_id)
    if age is None:
        continue

    participants.append((participant_id, t1_file, seg_file, age))

## Load existing graphs
if os.path.exists(checkpoint_path):
    participant_graphs = torch.load(checkpoint_path, weights_only=False)
else:
    participant_graphs = []

## Function to process a single participant
def process_participant(args):
    participant, t1_file, seg_file, age = args
    try:
        graph_builder = BrainGraph(seg_file, t1_file)
        graph_builder.get_region_stats()
        graph_builder.get_region_centroids()
        graph_builder.create_adjacency_list(k=5)
        graph_builder.get_brain_graph(age)
        graph = graph_builder.graph
        return (participant, graph)
    except Exception as e:
        return (participant, f"ERROR: {e}")
    finally:
        del graph_builder
        gc.collect()

## Parallel processing using 8 cores
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

chunk_size = 100
max_workers = 8

for batch in chunks(participants, chunk_size):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_participant, p): p[0] for p in batch}
        for future in tqdm(as_completed(futures), total=len(futures)):
            participant, result = future.result()
            if isinstance(result, str) and result.startswith("ERROR"):
                print(f"Failed for participant {participant}: {result}")
            else:
                participant_graphs.append(result)
                torch.save(participant_graphs, checkpoint_path)
                with open(completed_ids_path, 'a') as f:
                    f.write(f"{participant}\n")
