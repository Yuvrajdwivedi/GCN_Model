import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nclass, dropout): 
        super(GCN, self).__init__()
        
        # 2-Layer structural spatial processing
        self.gc1 = GraphConvolution(nfeat, nhid1) 
        self.gc2 = GraphConvolution(nhid1, nclass) 
        
        # The Node Identity Highway
        self.residual = nn.Linear(nfeat, nclass) 
        self.dropout = dropout
        
        # Initialize weights closely to prevent wild starting predictions
        nn.init.xavier_uniform_(self.residual.weight)

    def forward(self, x, adj): 
        # Path A: Spatial message aggregation
        gcn_path = self.gc1(x, adj)
        gcn_path = F.relu(gcn_path)
        gcn_path = F.dropout(gcn_path, self.dropout, training=self.training)
        gcn_path = self.gc2(gcn_path, adj) 
        
        # Path B: Direct feature path 
        skip_path = self.residual(x) 
        
        # Combine the structural signals
        return gcn_path + skip_path