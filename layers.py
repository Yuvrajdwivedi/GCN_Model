import math
import torch
from torch.nn.parameter import Parameter #it is a subclass of torch.Tensor that is used to define learnable parameters in a neural network. When you create a Parameter, it is automatically added to the list of parameters of the module it belongs to, and it will be updated during training.
from torch.nn.modules.module import Module #this is the base class for all neural network modules in PyTorch. It provides a way to define and manage layers, parameters, and forward passes in a structured manner.

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True): #this function initializes the GraphConvolution layer with the specified input and output features, and an optional bias term. It sets up the weight matrix and bias parameter (if applicable) and calls the reset_parameters method to initialize them.
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self): #this function initializes the weight and bias parameters of the GraphConvolution layer. It calculates a standard deviation based on the size of the weight matrix and uses it to uniformly initialize the weights and biases within a specific range. This helps in ensuring that the parameters start with reasonable values for training.
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj): #this function defines the forward pass of the GraphConvolution layer. 
        support = torch.mm(input, self.weight) #It takes an input feature matrix and an adjacency matrix as inputs. It first computes the support by multiplying the input features with the weight matrix. Then, it performs a sparse matrix multiplication between the adjacency matrix and the support to propagate information through the graph structure. 
        output = torch.spmm(adj, support) #If a bias term is present, it adds it to the output before returning the final result.
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'