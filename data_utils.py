import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(file_path="ncr_ride_bookings.csv"):
    """Loads graph structure and features from a single unified CSV file"""
    print(f'Loading dataset from single file: {file_path}...')
    # 1. Load the single unified CSV
    df = pd.read_csv(file_path)

    # 2. Extract Unique Nodes from your Source & Target columns
    # Assumes Column 0 is Source Node ID and Column 1 is Target Node ID
    source_nodes = df.iloc[:, 0].values
    target_nodes = df.iloc[:, 1].values
    
    # Get a sorted array of every unique node ID present in the dataset
    unique_nodes = np.unique(np.concatenate([source_nodes, target_nodes]))
    num_nodes = len(unique_nodes)
    
    # Map each unique raw ID to a clean matrix index matrix (0 to N-1)
    idx_map = {node_id: i for i, node_id in enumerate(unique_nodes)}

    # 3. Build Features and Labels per Node
    # Since features are row-by-row in the CSV, we group them by node ID 
    # to find the average feature values belonging to each specific location/node.
    
    # Select feature columns (Skipping cols 0, 1, and the 2nd col timestamp -> starting at index 3)
    feature_cols = df.columns[3:-1]
    df_numeric_features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    # We combine the Source Node IDs with the features to aggregate them
    df_features_mapped = pd.DataFrame(df_numeric_features)
    df_features_mapped['node_id'] = source_nodes
    
    # Group by node_id so we have exactly 1 row of features per unique node
    node_features_df = df_features_mapped.groupby('node_id').mean().reindex(unique_nodes, fill_value=0.0)
    features = sp.csr_matrix(node_features_df.values, dtype=np.float32)

    # Extract Labels (Aggregating the label per unique node, e.g., the most common label)
    df_labels = pd.DataFrame({'node_id': source_nodes, 'label': df.iloc[:, -1].values})
    node_labels_df = df_labels.groupby('node_id')['label'].agg(lambda x: x.mode()[0]).reindex(unique_nodes, fill_value=df.iloc[:, -1].mode()[0])
    labels = encode_onehot(node_labels_df.values)

    # 4. Build Graph Edges
    # Re-map the source and target columns using our 0-indexed map
    edges_mapped_source = np.array(list(map(idx_map.get, source_nodes)))
    edges_mapped_target = np.array(list(map(idx_map.get, target_nodes)))
    
    # Combine into an [Edge_Count, 2] array
    edges = np.vstack((edges_mapped_source, edges_mapped_target)).T
                     
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 5. Dynamic Data Splits
    # 1. Dynamically calculate the end points based on dataset size
    train_end = min(140, int(num_nodes * 0.1))
    val_end = min(500, int(num_nodes * 0.3))
    
    # 2. Convert features, labels, and adjacency matrix to PyTorch Tensors
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # 3. CORRECTED: Convert your dynamic ranges into LongTensors safely
    idx_train = torch.LongTensor(list(range(0, train_end)))
    idx_val = torch.LongTensor(list(range(train_end, val_end)))
    idx_test = torch.LongTensor(list(range(val_end, num_nodes)))

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)