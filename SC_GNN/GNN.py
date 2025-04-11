import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
import os
import pandas as pd
import numpy as np

class GNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.feature_maps = None
        self.gradients = None

    def forward(self, x, adj):
		# Layer 1
        x = torch.matmul(adj, x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.matmul(adj, x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Grad-CAM
        if x.requires_grad:
           x.retain_grad()  
           x.register_hook(self.capture_gradients)
        self.feature_maps = x
        
        # Mean pooling
        x = torch.mean(x, dim=0)

        # Output layer
        x = self.fc3(x)

        return x

	# Grad CAM architecture
    def get_attention_map(self):
        if self.feature_maps is None or self.gradients is None:
            raise ValueError("No feature maps or gradients captured.")
        weights = torch.mean(self.gradients, dim=0)  
        cam = torch.matmul(self.feature_maps, weights)
        return cam  

    def capture_gradients(self, grad):
        self.gradients = grad
        
# row wise norm of adj matrix        
def normalize_adjacency(adj):
    rowsum = adj.sum(1, keepdim=True)
    rowsum[rowsum == 0] = 1 
    return adj / rowsum

# Turns SC into graph and eliminate weak connections
def create_graph(matrix, age):
    num_nodes = matrix.shape[0]

    adj = torch.tensor(matrix, dtype=torch.float32)
    adj[adj < 0.01] = 0 
    adj = normalize_adjacency(adj)

    x = torch.eye(num_nodes)  
    
    return x, adj, torch.tensor([age], dtype=torch.float32)

# SC data loading function            
def load_matrices(data_path, ages_dict):
    graphs = []
    for file in os.listdir(data_path):
        if file.endswith("_connectivity.mat"):  
            subject_id = file.split("_connectivity.mat")[0]  
            if subject_id in ages_dict:
                mat_contents = sio.loadmat(os.path.join(data_path, file))
                if 'connectivity' in mat_contents:  
                    matrix = mat_contents['connectivity']  
                    graphs.append(create_graph(matrix, ages_dict[subject_id]))
                else:
                    print(f"Warning: 'and' variable not found in {file}")
            else:
                print(f"Warning: Subject ID {subject_id} not found in ages_dict")
    print(f"Total graphs loaded: {len(graphs)}")
    return graphs

