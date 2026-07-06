import os
import torch
import numpy as np
from models import GCN
from data_utils import load_data

def run_inference():
    print("Initializing Forecasting Engine...")

    # 1. Load the network layout and a snapshot of current data
    adj, features, _, _ = load_data()
    num_nodes = features.shape[0]
    n_input_features = features.shape[1]

    # 2. Rebuild the model architecture and load your saved weights
    model = GCN(nfeat=n_input_features, nhid1=32, nclass=1, dropout=0.0)
    
    # Absolute path defense to make sure it always finds the saved weights
    model_path = r"C:\Users\yuvra\OneDrive\YUVRAJ FILES\Coding\UberRidesNCR_GCN_Model\gcn_traffic_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CRITICAL: The model weights file is missing at: {model_path}\nPlease run 'python train.py' to generate it first.")
        
    model.load_state_dict(torch.load(model_path))
    
    # Lock the model into evaluation mode (turns off dropout, locks weights)
    model.eval()

    print("\nExecuting Spatio-Temporal Forecast...")
    
    # 3. Run the forward pass without tracking gradients (saves memory/time)
    with torch.no_grad():
        future_predictions = model(features, adj)

    # 4. Define Traffic Detection Thresholds
    # Traffic is flagged if the predicted speed drops below 35 mph, 
    # OR if it drops by more than 15 mph compared to what it is right now.
    TRAFFIC_SPEED_THRESHOLD = 35.0 
    
    print(f"\n{'='*75}")
    print("LIVE TRAFFIC ANOMALY & CONGESTION RADAR:")
    print(f"{'Sensor':<10} | {'Current Speed':<15} | {'Predicted Speed':<15} | {'Traffic Status':<15}")
    print(r"-"*75)
    
    traffic_zones_count = 0

    # Scan all nodes across your network layout
    for sensor_id in range(num_nodes):
        # Calculate current historical snapshot speed for this sensor node
        current_avg_speed = torch.mean(features[sensor_id]).item()
        predicted_speed = future_predictions[sensor_id].item()
        
        # Core Rule: If predicted speed is dangerously low or falls off a cliff
        if predicted_speed < TRAFFIC_SPEED_THRESHOLD or (current_avg_speed - predicted_speed) > 15.0:
            status = "🔴 TRAFFIC DETECTED"
            traffic_zones_count += 1
        elif predicted_speed < 45.0:
            status = "🟡 SLOWING DOWN"
        else:
            status = "🟢 FREE FLOW"
            
        # Display the first 15 sensors on your terminal as a real-time monitor panel
        if sensor_id < 15:
            print(f"Sensor {sensor_id:03d} | {current_avg_speed:.1f} mph       | {predicted_speed:.1f} mph       | {status}")
            
    print(f"{'='*75}")
    print(f"SYSTEM SUMMARY: Scanned {num_nodes} nodes. Found {traffic_zones_count} active traffic alerts.")
    print(f"{'='*75}")

if __name__ == "__main__":
    run_inference()