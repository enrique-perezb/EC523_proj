import nibabel as nib
import numpy as np
from skimage import measure
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

class BrainGraph:

    def __init__(self, seg_nii, t1_nii):
        self.seg_img = nib.load(seg_nii)
        self.seg_data = self.seg_img.get_fdata()
        self.voxel_volume = np.prod(np.abs(self.seg_img.header.get_zooms()))
        self.affine = self.seg_img.affine

        t1_img = nib.load(t1_nii)
        self.t1_data = t1_img.get_fdata()

        self.labels = None
        self.label_list = None
        self.region_stats = None
        self.region_centroids = None
        self.edge_index = None
        self.graph = None

    def get_region_stats(self):
        region_stats = {}
        labels = np.unique(self.seg_data)
        labels = labels[labels != 0]

        for label in labels:
            region_stats[label] = {}
            mask = self.seg_data == label

            # Intensity-based stats
            region_intensity = self.t1_data[mask]
            region_stats[label]["mean_intensity"] = np.mean(region_intensity)
            region_stats[label]["variance_intensity"] = np.var(region_intensity)

            # Volume
            region_volume = np.sum(mask) * self.voxel_volume
            region_stats[label]["volume_mm3"] = region_volume

            # Surface area for thickness estimation
            verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0)
            spacing = self.seg_img.header.get_zooms()
            try:
                surface_area = measure.mesh_surface_area(verts * spacing, faces)
                thickness = region_volume / surface_area if surface_area > 0 else 0
            except Exception:
                thickness = 0

            region_stats[label]["approx_thickness"] = thickness

        self.labels = labels
        self.label_list = sorted(region_stats.keys())
        self.region_stats = region_stats

    def get_region_centroids(self):
        region_centroids = {}
        for label in self.labels:
            mask = self.seg_data == label
            coords = np.column_stack(np.where(mask))
            if coords.size == 0:
                continue
            world_coords = nib.affines.apply_affine(self.affine, coords)
            centroid = world_coords.mean(axis=0)
            region_centroids[label] = centroid
        self.region_centroids = region_centroids

    def create_adjacency_list(self, k=5):
        centroids = np.array([self.region_centroids[label] for label in self.label_list])
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(centroids)
        _, indices = nbrs.kneighbors(centroids)

        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:
                edge_list.append((i, j))

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def get_brain_graph(self, age, group=None):
        features = []
        for label in self.label_list:
            stats = self.region_stats[label]
            feature_vector = [
                stats.get("mean_intensity", 0),
                stats.get("variance_intensity", 0),
                stats.get("volume_mm3", 0),
                stats.get("approx_thickness", 0)
            ]
            features.append(feature_vector)

        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(age, dtype=torch.float)
        self.graph = Data(x=x, edge_index=self.edge_index, y=y)

        if group is not None:
            self.graph.group = group
