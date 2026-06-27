import torch.nn as nn #this line imports the torch.nn module, which provides classes and functions to build neural networks in PyTorch. It includes various layers, loss functions, and utilities for constructing and training models.
import torch.nn.functional as F #it provides a functional interface for various operations in PyTorch, including activation functions, loss functions, and other utilities. It allows you to apply these operations directly to tensors without needing to define a separate layer or module.
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout): #this function initializes the GCN model with the specified number of input features (nfeat), hidden units (nhid), output classes (nclass), and dropout rate (dropout). It sets up two GraphConvolution layers and a dropout layer to be used in the forward pass.
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nhid3)  # Transforms nhid2 -> nhid3
        self.gc4 = GraphConvolution(nhid3, nclass) # New final layer: nhid3 -> nclass
        self.dropout = dropout

    def forward(self, x, adj): #this function defines the forward pass of the GCN model. It takes input features (x) and adjacency matrix (adj) as inputs. The input features are passed through the first GraphConvolution layer followed by ReLU activation and dropout. Then, they are passed through the second GraphConvolution layer followed by ReLU activation and dropout. Finally, the output is passed through the third GraphConvolution layer to produce the final predictions.
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        return F.log_softmax(x, dim=1) #the final output is passed through F.log_softmax to convert the logits into log probabilities for each class. The dim=1 argument specifies that the softmax should be applied along the class dimension.
