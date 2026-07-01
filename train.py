import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold

from data_utils import load_data
from models import GCN

# 1. Config
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.3)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 2. Load the Local Data
adj, features, labels, feature_cols = load_data()
num_nodes = features.shape[0]
n_input_features = features.shape[1] 

# 🌟 GRAPH TELEMETRY: Calculate and display the exact graph density
num_edges = adj._nnz()
max_possible_edges = num_nodes * num_nodes
graph_density = (num_edges / max_possible_edges) * 100

print(f"\n{'='*40}")
print(f"GRAPH ARCHITECTURE SUMMARY:")
print(f"Total Nodes: {num_nodes}")
print(f"Active Network Edges: {num_edges}")
print(f"Network Density: {graph_density:.2f}%")
print(f"{'='*40}\n")

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

def train_epoch(model, optimizer, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.mse_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss_train.item()

def validate_epoch(model, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.mse_loss(output[idx_val], labels[idx_val])
        mae_val = torch.mean(torch.abs(output[idx_val] - labels[idx_val]))
        
        y_true = labels[idx_val]
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - output[idx_val]) ** 2)
        acc_val = 1 - (ss_res / (ss_tot + 1e-8))
    return loss_val.item(), mae_val.item(), acc_val.item()

def test_fold(model, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.mse_loss(output[idx_test], labels[idx_test])
        mae_test = torch.mean(torch.abs(output[idx_test] - labels[idx_test]))
        
        y_true = labels[idx_test]
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - output[idx_test]) ** 2)
        acc_test = 1 - (ss_res / (ss_tot + 1e-8))
    return loss_test.item(), mae_test.item(), acc_test.item()

kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
cv_accuracies = []
cv_maes = []

print(f"Executing 5-Fold Cross-Validation splits...")

for fold, (train_and_val_indices, test_indices) in enumerate(kf.split(np.arange(num_nodes))):
    print(f"\n--- FOLD {fold + 1} / 5 ---")
    
    np.random.shuffle(train_and_val_indices)
    split_point = int(0.85 * len(train_and_val_indices))
    
    idx_train = torch.LongTensor(train_and_val_indices[:split_point])
    idx_val = torch.LongTensor(train_and_val_indices[split_point:])
    idx_test = torch.LongTensor(test_indices)
    
    if args.cuda:
        idx_train, idx_val, idx_test = idx_train.cuda(), idx_val.cuda(), idx_test.cuda()
        
    model = GCN(nfeat=n_input_features, nhid1=args.hidden, nclass=1, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.cuda:
        model.cuda()
        
    for epoch in range(args.epochs):
        t_start = time.time()
        loss_t = train_epoch(model, optimizer, idx_train)
        loss_v, mae_v, acc_v = validate_epoch(model, idx_val)
        
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1:03d} | Train Loss: {loss_t:.4f} | Val Loss: {loss_v:.4f} | Val Acc(R²): {acc_v:.4f}')
            
    loss_te, mae_te, acc_te = test_fold(model, idx_test)
    print(f">> Fold {fold + 1} Test Results -> MSE Loss: {loss_te:.4f} | MAE: {mae_te:.4f} | Accuracy (R²): {acc_te:.4f}")
    
    cv_maes.append(mae_te)
    cv_accuracies.append(acc_te)

print(f"\n{'='*60}")
print(f"FINAL METR-LA BENCHMARK K-FOLD SUMMARY:")
print(f"Average Test MAE: {np.mean(cv_maes):.4f} (+/- {np.std(cv_maes):.4f})")
print(f"Average Test Accuracy (Global R² Score): {np.mean(cv_accuracies):.4f}")
print(f"{'='*60}")
# --- STEP 8: SAVE THE FINAL TRAINED MODEL ---
print("\nSaving final model weights...")
torch.save(model.state_dict(), "gcn_traffic_model.pth")
print("Model saved successfully as 'gcn_traffic_model.pth'!")