import torch
import numpy as np
from models import GCN
from data_utils import load_data

def run_inference():
    print("Initializing Forecasting Engine...")

    # 1. Load the network layout and a snapshot of current data
    # (In a real enterprise system, 'features' would be a live stream from physical sensors)
    adj, features, _, _ = load_data()
    num_nodes = features.shape[0]
    n_input_features = features.shape[1]

    # 2. Rebuild the model architecture and load your saved weights
    model = GCN(nfeat=n_input_features, nhid1=32, nclass=1, dropout=0.0)
    model.load_state_dict(torch.load("gcn_traffic_model.pth"))
    
    # Lock the model into evaluation mode (turns off dropout, locks weights)
    model.eval()

    print("\nExecuting Spatio-Temporal Forecast...")
    
    # 3. Run the forward pass without tracking gradients (saves memory/time)
    with torch.no_grad():
        # We pass the current speeds (features) and the physical map (adj)
        future_predictions = model(features, adj)

    # 4. Output the results
    print(f"\n{'='*40}")
    print("LIVE PREDICTION RESULTS (Next Time Step):")
    
    # Let's look at the forecasted speeds for the first 5 highway sensors
    for sensor_id in range(5):
        current_avg_speed = torch.mean(features[sensor_id]).item()
        predicted_speed = future_predictions[sensor_id].item()
        
        print(f"Sensor {sensor_id:03d} | Current Avg: {current_avg_speed:.1f} mph | FORECAST: {predicted_speed:.1f} mph")
    
    print(f"{'='*40}")

if __name__ == "__main__":
    run_inference()