import torch 
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from GNN import GNN, load_matrices

# === Paths ===
DATA_PATH = "/projectnb/ec523/projects/proj_GS_LQ_EPB/data/SC_Matricies/ADNI/tractography/"
AGE_FILE = "/projectnb/ec523/projects/proj_GS_LQ_EPB/data/SC_Matricies/ADNI/ages.csv"
MODEL_PATH = "/projectnb/ec523/projects/proj_GS_LQ_EPB/models/SC_GNN/combTest/training/models/20250415_133201/best_model_fold_10.pt"
ROI_FILE = "brainnetome_rois.txt"  # Add path if needed

# === Results directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"testing/results/fine_tune_ADNI_eval_{timestamp}/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load and normalize age data ===
ages_df = pd.read_csv(AGE_FILE)
ages_df['subject_id'] = ages_df['subject_id'].astype(str).str.strip()

# Min-max normalization
min_age = ages_df['age'].min()
max_age = ages_df['age'].max()
ages_df['age_norm'] = (ages_df['age'] - min_age) / (max_age - min_age)
ages_dict = dict(zip(ages_df['subject_id'], ages_df['age_norm']))

# === Load test graphs ===
test_graphs = load_matrices(DATA_PATH, ages_dict)
subject_ids = list(ages_dict.keys())

# === Load ROI names ===
with open(ROI_FILE, 'r') as f:
    rois = [line.strip() for line in f.readlines()]

# === Load model ===
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = test_graphs[0][0].shape[1]
model = GNN(num_features=num_features, hidden_dim=32, output_dim=1).to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()

# === Inference and Evaluation ===
true_vals, pred_vals, mae_losses, mse_losses, subject_names = [], [], [], [], []

for idx, (x, adj, y) in enumerate(test_graphs):
    subj_id = subject_ids[idx]
    x, adj, y = x.to(dev), adj.to(dev), y.to(dev)
    output = model(x, adj).squeeze()

    # Denormalize
    y_true = y.item() * (max_age - min_age) + min_age
    y_pred = output.item() * (max_age - min_age) + min_age

    subject_names.append(subj_id)
    true_vals.append(y_true)
    pred_vals.append(y_pred)
    mae_losses.append(abs(y_pred - y_true))
    mse_losses.append((y_pred - y_true) ** 2)

# === Metrics ===
mae = np.mean(mae_losses)
mse = np.mean(mse_losses)
rmse = np.sqrt(mse)

print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# === Save results with subject IDs ===
results_df = pd.DataFrame({
    "Subject ID": subject_names,
    "True Age": true_vals,
    "Predicted Age": pred_vals,
    "MAE": mae_losses,
    "MSE": mse_losses
})
results_df.to_csv(os.path.join(RESULTS_DIR, "test_results.csv"), index=False)

# === Grad-CAM Analysis ===
x, adj, _ = test_graphs[0]
x, adj = x.to(dev), adj.to(dev)
output = model(x, adj).squeeze()
model.zero_grad()
output.backward()
attention_map = model.get_attention_map()

sorted_rois = sorted(zip(rois, attention_map.detach().cpu().numpy()), key=lambda x: x[1], reverse=True)

# Save top 10 contributing ROIs
with open(os.path.join(RESULTS_DIR, "top_10_rois.txt"), "w") as f:
    f.write("Top 10 Contributing ROIs:\n")
    for roi, weight in sorted_rois[:10]:
        f.write(f"{roi}: {weight:.4f}\n")

# === Plotting ===
plt.figure(figsize=(8, 6))
plt.scatter(true_vals, pred_vals, alpha=0.7, color='dodgerblue')
plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title(f"Predicted vs True Age (ADNI Test Set) - MAE: {mae:.4f}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pred_vs_true.png"))
plt.show()
