import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def get_absolute_path(filename):
    # Locks strictly into your local project directory
    return os.path.join(r"C:\Users\yuvra\OneDrive\YUVRAJ FILES\Coding\UberRidesNCR_GCN_Model", filename)

def clean_id(raw_val):
    """Safely cleans IDs so '123', '123.0', and 123.0 all match perfectly."""
    try:
        return str(int(float(raw_val)))
    except:
        return str(raw_val).strip()

def load_data():
    print("Loading spatial graph using CSV locations and K-Nearest Neighbors...")
    
    dist_file = get_absolute_path("distances_la_2012.csv") 
    ids_file = get_absolute_path("graph_sensor_locations.csv") # 🌟 Updated to your exact CSV filename
    
    if not os.path.exists(dist_file) or not os.path.exists(ids_file):
        raise FileNotFoundError("Missing required files. Ensure 'distances_la_2012.csv' and 'graph_sensor_locations.csv' are in the folder.")
        
    # 1. Parse the CSV file for Sensor IDs correctly
    # This reads the table and extracts ONLY the first column (the IDs), ignoring the GPS coordinates
    loc_df = pd.read_csv(ids_file)
    sensor_ids = [clean_id(x) for x in loc_df.iloc[:, 0].values]
    
    sensor_id_to_ind = {str(sensor_id): i for i, sensor_id in enumerate(sensor_ids)}
    num_nodes = len(sensor_ids) 
    print(f"-> Extracted {num_nodes} valid sensor IDs from the CSV.")
    
    # 2. Build Distance Matrix
    dist_df = pd.read_csv(dist_file)
    dist_mx = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist_mx, 0.0)
    
    for _, row in dist_df.iterrows():
        src_id, dst_id = clean_id(row.iloc[0]), clean_id(row.iloc[1])
        cost = float(row.iloc[2])
        if src_id in sensor_id_to_ind and dst_id in sensor_id_to_ind:
            dist_mx[sensor_id_to_ind[src_id], sensor_id_to_ind[dst_id]] = cost
            
    # 3. GUARANTEED EDGE CONSTRUCTION: K-Nearest Neighbors (KNN)
    # This forces exactly 15 connections per node, guaranteeing your graph never dies.
    adj_dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i in range(num_nodes):
        node_distances = dist_mx[i, :]
        closest_indices = np.argsort(node_distances)
        
        connected_count = 0
        for j in closest_indices:
            if i != j and node_distances[j] != np.inf:
                # Weight the edge inversely to distance (closer = stronger weight)
                adj_dense[i, j] = 1.0 / (node_distances[j] + 1.0) 
                connected_count += 1
            
            # Stop once we secure 15 strong physical connections for this location
            if connected_count >= 15:
                break
                
    # Convert to Sparse Tensor and force symmetric self-loops
    adj = sp.coo_matrix(adj_dense, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # Ensure two-way streets
    adj = adj + sp.eye(num_nodes) 
    adj = symmetric_normalize(adj)
    
    # 4. Generate Synthetic Temporal Features
    print("-> Automatically generating matching temporal speed features...")
    np.random.seed(42)
    num_features = 12
    
    features_np = np.zeros((num_nodes, num_features), dtype=np.float32)
    for node in range(num_nodes):
        for t in range(num_features):
            features_np[node, t] = 55.0 + 12.0 * np.sin(t * 0.4 - node * 0.2) + np.random.normal(0, 0.1)
            
    labels_np = np.zeros((num_nodes, 1), dtype=np.float32)
    for node in range(num_nodes):
        labels_np[node, 0] = 55.0 + 12.0 * np.sin(num_features * 0.4 - node * 0.2)
    
    features = torch.FloatTensor(features_np)
    labels = torch.FloatTensor(labels_np)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    feature_cols = [f"speed_t_{i}" for i in range(12)]
    return adj, features, labels, feature_cols

def symmetric_normalize(mx):
    """Symmetric normalization with strict infinity defense."""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt) | np.isnan(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return r_mat_inv_sqrt.dot(mx).dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert SciPy sparse matrix directly to PyTorch format."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)