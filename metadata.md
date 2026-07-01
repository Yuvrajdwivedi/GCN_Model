# METR-LA Traffic Graph Dataset Metadata

This document serves as the official metadata registry and operational guide for the traffic sensor network graph assets of the METR-LA dataset found in the [METR-LA Dataset Assets](https://github.com/deepkashiwa20/DL-Traff-Graph/blob/main/METRLA/adj_mx.pkl) repository. These files define the spatial structure of a 207-node highway loop detector network in Los Angeles County, used primarily for spatial-temporal traffic forecasting models.

---

## File Registry

| File Name | Format | Dimensions / Shape | Key Fields / Columns | Functional Description |
| :--- | :--- | :--- | :--- | :--- |
| **`graph_sensor_locations.csv`** | CSV (Text) | 207 rows × 4 columns | `index`, `sensor_id`, `latitude`, `longitude` | Coordinates mapping. Links the internal matrix index (0 to 206) to real-world Caltrans PeMS IDs and GPS coordinates. |
| **`distances_la_2012.csv`** | CSV (Text) | Variable directed pairs × 3 columns | `from`, `to`, `cost` | Physical network topology. Lists directed graph edges representing actual freeway distances (travel cost) between sensors. |
| **`W_metrla.csv`** | CSV (Text) | 207 rows × 207 columns | Row/Column Headers (Sensor IDs/Indices) | Pre-computed spatial weight matrix. Physical distances are mathematically transformed into localized spatial correlation coefficients. |
| **`adj_mx.pkl`** | Pickle (Binary) | Python Tuple: `(list, dict, np.ndarray)` | `sensor_ids`, `sensor_id_to_ind`, `adj_mx` | Machine-learning-ready package. Bundles the string sensor IDs, index map, and weight array into a single binary object for rapid loading. |

---

## Data Architecture & Integration Pipeline

When training a Spatial-Temporal Graph Neural Network (GNN), the data flows sequentially through these files to construct the graph structure that constrains neural message passing.

### 1. Geographical Anchor (`graph_sensor_locations.csv`)
* **Role:** Establishes the real-world ground truth.
* **Mechanism:** It maps the sequence order of traffic metrics to explicit geospatial markers. This is critical for computing spatial distances or when plotting traffic state predictions onto a visualization dashboard or interactive GIS map.

### 2. Network Topology (`distances_la_2012.csv`)
* **Role:** Represents physical connectivity.
* **Mechanism:** Road networks are constrained topologies (cars cannot fly over blocks to adjacent highways). This file stores the directed network path lengths from a specific upstream node (`from`) to a downstream node (`to`). If a sensor pair is missing from this list, it implies no significant direct causal traffic relationship exists between them.

### 3. Numerical Conditioning (`W_metrla.csv`)
* **Role:** Fuses and normalizes distances into an edge-weight model.
* **Mechanism:** GNNs cannot effectively interpret raw distance values directly (e.g., 5 miles vs 0.2 miles) because neural weights perform better when strong relationships have values scaling towards 1.0 and distant ones decay toward 0. To achieve this, raw distances ($d$) are converted using a thresholded Gaussian kernel:
  
  W_ij = exp(- (distance(v_i, v_j) / sigma)^2)
  
* **Truncation:** If the computed weight falls below a given epsilon threshold, it is forcibly truncated to `0.0`. This results in a sparse matrix that ignores weak, distant noise and drastically reduces memory overhead during graph convolutions.

### 4. Direct PyTorch/TensorFlow Ingestion (`adj_mx.pkl`)
* **Role:** Eliminates serialization and parsing latency.
* **Mechanism:** Parsing plaintext CSVs at runtime bottlenecks training. The pickle file acts as a compiled Python dictionary/tuple. It ensures the matrix index exactly matches the spatial dimensions of your historical speed tensor arrays.

---

## How to Load the Dataset Assets in Python

The binary `.pkl` asset cannot be parsed as plain text. Use the following code block to unpickle and inspect the packaged components directly within your training pipeline:

```python
import pickle
import numpy as np

# Load the serialized metadata package
with open('adj_mx.pkl', 'rb') as f:
    try:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)
    except UnicodeDecodeError:
        # Fallback for cross-compatible python 2/3 encoding
        f.seek(0)
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')

# Validate contents
print(f"Total Configured Sensors: {len(sensor_ids)}")
print(f"Lookup Map Sample (Sensor to Tensor Index): {list(sensor_id_to_ind.items())[:3]}")
print(f"Adjacency Matrix Array Shape: {adj_mx.shape}")
print(f"Matrix Sparsity Ratio: {np.mean(adj_mx == 0):.2%}")