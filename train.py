from __future__ import division #this is to ensure that the division operator behaves like in Python 3, even if we are using Python 2. It ensures that the division of two integers results in a float, rather than an integer.
from __future__ import print_function #this is to ensure that the print function behaves like in Python 3, even if we are using Python 2. It allows us to use the print function with parentheses, rather than the print statement without parentheses.

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from data_utils import load_data
from eval_utils import accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
# Replace the single '--hidden' argument with these:
parser.add_argument('--hidden1', type=int, default=32, help='Number of hidden units for layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of hidden units for layer 2.')
parser.add_argument('--hidden3', type=int, default=8, help='Number of hidden units for layer 3.') # New argument for the third hidden layer
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid1=args.hidden1,  # First hidden layer size
            nhid2=args.hidden2,  # Second hidden layer size
            nhid3=args.hidden3,  # Third hidden layer size (New Layer)
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Isolate validation to prevent variable overwrites
    model.eval()
    with torch.no_grad():
        output_val = model(features, adj)
    
    loss_val = F.nll_loss(output_val[idx_val], labels[idx_val])
    acc_val = accuracy(output_val[idx_val], labels[idx_val])
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    with torch.no_grad():
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),  
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
train_set = set(idx_train.cpu().numpy())
test_set = set(idx_test.cpu().numpy())
val_set = set(idx_val.cpu().numpy())
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()