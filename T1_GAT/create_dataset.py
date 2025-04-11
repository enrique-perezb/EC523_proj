from torch_geometric.data import Dataset


## Helper class to format the graphs into hte structure I need
class BrainGraphDataset(Dataset):
    def __init__(self, graph_list):
        super().__init__()
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
