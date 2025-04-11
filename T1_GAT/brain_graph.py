import nibabel as nib
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

class BrainGraph:

    # Load image and initialize needed variables
    def __init__(self, seg_nii, ct_nii):
        self.seg_img = nib.load(seg_nii)
        self.seg_data = self.seg_img.get_fdata()
        self.voxel_volume = np.prod(np.abs(self.seg_img.header.get_zooms()))

        ct_img = nib.load(ct_nii)
        self.ct_data = ct_img.get_fdata()

        self.labels = None
        self.label_list = None
        self.region_stats = None
        self.region_centroids = None
        self.edge_index = None
        self.graph = None

    # Go through each region and extract stats(voxel count, volume, mean cortical thickness)
    def get_region_stats(self):
        region_stats = {}
        labels = np.unique(self.seg_data)
        labels = labels[labels != 0]

        for label in labels:
            region_stats[label] = {}

            mask = self.seg_data == label
            region_ct_vals = self.ct_data[mask]
            region_ct_vals = region_ct_vals[region_ct_vals > 0]

            region_stats[label]['voxel_count'] = np.sum(mask)
            region_stats[label]['volume_mm3'] = region_stats[label]['voxel_count'] * self.voxel_volume
            region_stats[label]['mean_thickness'] = np.mean(region_ct_vals) if region_ct_vals.size > 0 else 0

        self.labels = labels
        self.label_list = sorted(region_stats.keys())
        self.region_stats = region_stats

    # Find the centroid of each region (used for adjacency and graph creation)
    def get_region_centroids(self):
        region_centroids = {}

        for label in self.labels:
            mask = self.seg_data == label
            coords = np.column_stack(np.where(mask))
            if coords.size == 0:
                continue

            world_coords = nib.affines.apply_affine(self.seg_img.affine, coords)
            centroid = world_coords.mean(axis=0)
            region_centroids[label] = centroid

        self.region_centroids = region_centroids

    # Finds the 5 nearest neighbors (centroids) to create edges
    def create_adjacency_list(self, k=5):
        centroids = np.array([self.region_centroids[label] for label in self.label_list])

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
        _, indices = nbrs.kneighbors(centroids)

        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:
                edge_list.append((i, j))

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Puts region stats into nodes and creates edge matrix. Puts it into torch_geometric Data object for easier future use.
    def get_brain_graph(self, age):
        features = []
        for label in self.label_list:
            stats = self.region_stats[label]
            feature_vector = [
                stats.get('mean_thickness', 0),
                stats.get('volume_mm3', 0),
                stats.get('voxel_count', 0)
            ]
            features.append(feature_vector)

        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(age, dtype=torch.float)

        self.graph = Data(x=x, edge_index=self.edge_index, y=y)
