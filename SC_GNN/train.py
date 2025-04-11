import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from GNN import GNN, load_matrices
from datetime import datetime

# Hyperparameters and directories
DATA_PATH = "/projectnb/ec523/projects/proj_GS_LQ_EPB/data/SC_Matricies/HCP/tractography/"
AGE_FILE = "/projectnb/ec523/projects/proj_GS_LQ_EPB/data/SC_Matricies/HCP/age/ages.csv"
BATCH_SIZE = 8
HIDDEN_DIM = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"training/logs/{timestamp}"
MODEL_DIR = f"training/models/{timestamp}"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "training_log.txt")

# Load data
ages_df = pd.read_csv(AGE_FILE)
ages_dict = {str(row['subject_id']).strip(): row['age'] for _, row in ages_df.iterrows()}
graphs = load_matrices(DATA_PATH, ages_dict)

# cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_rmse_per_fold = []
best_mae_per_fold = []

# Load ROI names
with open('brainnetome_rois.txt', 'r') as f:
    rois = [line.strip() for line in f.readlines()]

for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    # Initialize model
    model = GNN(num_features=train_graphs[0][0].shape[1], hidden_dim=HIDDEN_DIM, output_dim=1).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #optimizer
    criterion = torch.nn.MSELoss() #loss function

    best_loss = float('inf')
    best_mae = float('inf')
    patience = 0

    with open(log_file, "a") as log:
        log.write(f"\nFold {fold + 1} Training\n")

    for epoch in range(EPOCHS):
		# training
        model.train()
        train_losses = []		
        for data in train_graphs:
            x, adj, y = data
            x, adj, y = x.to(dev), adj.to(dev), y.to(dev)

            optimizer.zero_grad()
            output = model(x, adj).view(-1)
            loss = criterion(output, y.view(-1))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        
		# validation
        model.eval()
        val_loss = []
        val_mae = []
        with torch.no_grad():
            for data in test_graphs:
                x, adj, y = data
                x, adj, y = x.to(dev), adj.to(dev), y.to(dev)
                output = model(x, adj).view(-1)
                val_loss.append(F.mse_loss(output, y.view(-1)).item())
                val_mae.append(F.l1_loss(output, y.view(-1)).item())

        avg_val_loss = np.mean(val_loss)
        avg_val_mae = np.mean(val_mae)

        with open(log_file, "a") as log:
            log.write(
                f"Epoch {epoch + 1}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation RMSE: {np.sqrt(avg_val_loss):.4f}, "
                f"Validation MAE: {avg_val_mae:.4f}\n"
            )
		# Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_mae = avg_val_mae
            patience = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_model_fold_{fold + 1}.pt"))
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                with open(log_file, "a") as log:
                    log.write("Early stopping triggered\n")
                break

    fold_best_rmse = np.sqrt(best_loss)
    best_rmse_per_fold.append(fold_best_rmse)
    best_mae_per_fold.append(best_mae)

    print(f'Fold {fold + 1}, Best Validation RMSE: {fold_best_rmse:.4f}, Best Validation MAE: {best_mae:.4f}')

    # Grad CAM
    x, adj, y = test_graphs[0]
    x, adj = x.to(dev), adj.to(dev)
    output = model(x, adj).squeeze()
    model.zero_grad()
    output.backward()
    attention_map = model.get_attention_map()

    sorted_rois = sorted(zip(rois, attention_map.detach().cpu().numpy()), key=lambda x: x[1], reverse=True)

    with open(log_file, "a") as log:
        log.write(f"Top contributing ROIs for Fold {fold + 1}:\n")
        for roi, weight in sorted_rois[:10]:
            log.write(f"{roi}: {weight:.4f}\n")

# Plot error
plt.figure(figsize=(10, 6))
folds = list(range(1, len(best_rmse_per_fold) + 1))
plt.plot(folds, best_rmse_per_fold, marker='o', label='Best RMSE per Fold')
plt.plot(folds, best_mae_per_fold, marker='s', label='Best MAE per Fold')
plt.xlabel("Fold")
plt.ylabel("Loss")
plt.title("Best RMSE and MAE per Fold")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(LOG_DIR, "per_fold_best_loss_plot.png"))
plt.show()

# Log final metrics
with open(log_file, "a") as log:
    log.write("\n=== Summary of Best Validation Metrics per Fold ===\n")
    for i, (rmse, mae) in enumerate(zip(best_rmse_per_fold, best_mae_per_fold), start=1):
        log.write(f"Fold {i}: RMSE = {rmse:.4f}, MAE = {mae:.4f}\n")
