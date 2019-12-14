from data.data_processtor import processtor
from dgl.nn.pytorch.conv import GATConv
import numpy as np
import torch
import logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H-%M-%S %p'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

users, items, graph = processtor('ml-100k')

# init GAT model.
gat = GATConv(in_feats=91, out_feats=32, num_heads=2)


input = torch.from_numpy(items).float()

output = gat(graph, input)

print(output.size())