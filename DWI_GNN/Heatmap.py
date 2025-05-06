#!/usr/bin/env python
"""
make_roi_heatmap.py
-------------------
Create a 3-D heat-map of ROI importance scores in Brainnetome space.

Simply update the three PATH CONSTANTS below and run:

    python make_roi_heatmap.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting

#                                                                 
# >>> EDIT THESE THREE PATH CONSTANTS TO MATCH YOUR ENVIRONMENT <<<

ATLAS_PATH   = "/projectnb/ec523/projects/proj_GS_LQ_EPB/models/SC_GNN/combTest/BN_Atlas_246_2mm.nii.gz"   # integer-label Brainnetome atlas
ROI_CSV_PATH = "/projectnb/ec523/projects/proj_GS_LQ_EPB/models/SC_GNN/combTest/testing/results/fine_tune_AD_eval_20250423_160051/importance.csv"
OUT_PREFIX   = "age_roi_heatmap_AD"                    # output basename (no extension)

#                                                                 

def load_roi_table(csv_path: str) -> pd.DataFrame:
    """Read ROI table and sanity-check required columns."""
    df = pd.read_csv(csv_path)
    required = {"atlas_index", "importance"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}")
    return df

def build_heatmap(atlas_img: nib.Nifti1Image, roi_df: pd.DataFrame) -> nib.Nifti1Image:
    """Create a volume whose voxels equal the importance of their ROI label."""
    atlas_data = atlas_img.get_fdata().astype(int)
    heat = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in roi_df.iterrows():
        label = int(row["atlas_index"])
        weight = float(row["importance"])
        heat[atlas_data == label] = weight

    return nib.Nifti1Image(heat, affine=atlas_img.affine, header=atlas_img.header)

def save_and_plot(heat_img: nib.Nifti1Image, prefix: str):
    """Write NIfTI file and create a glass-brain snapshot for QC."""
    nii_path = f"{prefix}.nii.gz"
    png_path = f"{prefix}_glass.png"
    nib.save(heat_img, nii_path)

    display = plotting.plot_glass_brain(
        heat_img,
        colorbar=True,
        plot_abs=False,
        threshold=1e-6,
        title="Top-ROI importance (Brainnetome) AD",
    )
    display.savefig(png_path, dpi=300)
    display.close()

    print(f"Saved NIfTI : {nii_path}")
    print(f"Saved PNG   : {png_path}")

def main():
    if not os.path.exists(ATLAS_PATH):
        raise FileNotFoundError(f"Atlas not found: {ATLAS_PATH}")
    if not os.path.exists(ROI_CSV_PATH):
        raise FileNotFoundError(f"ROI CSV not found: {ROI_CSV_PATH}")

    atlas_img = nib.load(ATLAS_PATH)
    roi_df = load_roi_table(ROI_CSV_PATH)
    heat_img = build_heatmap(atlas_img, roi_df)
    save_and_plot(heat_img, OUT_PREFIX)

if __name__ == "__main__":
    main()
