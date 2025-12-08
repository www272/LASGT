import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv

from GATELayer import *
# flags = tf.app.flags
# FLAGS = flags.FLAGS

class GATE(nn.Module):
    def __init__(self, attribute_number, hidden_size, embedding_size, alpha):
        super(GATE, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        # self.conv1 = GATELayer(attribute_number, hidden_size, alpha)
        # self.conv2 = GATELayer(hidden_size, embedding_size, alpha)

        self.conv = GATELayer(attribute_number,embedding_size,alpha)
        self.conv1 = MultiHeadGATLayer(4, attribute_number, embedding_size, alpha, concat=False)
        # self.conv2 = MultiHeadGATLayer(6, embedding_size*6,   attribute_number,  alpha, concat=False)
        # self.conv1 = MultiHeadGATE(attribute_number, embedding_size, alpha, heads=6, concat=True)
        # self.conv1 = GATConv(attribute_number, embedding_size,1)
    def forward(self, x, adj, M,adj_attr):
        # h1 = self.conv1(x, adj, M)
        # h2 = self.conv1(x,adj_attr,M)

        h1 = self.conv1(x, adj,M)
        h2 = self.conv1(x, adj_attr,M)


        z = (h1 + h2) / 2

        z = F.normalize(z, p=2, dim=1)

        z1 = F.normalize(h1, p=2, dim=1)
        z2 = F.normalize(h2, p=2, dim=1)

        A_pred = self.dot_product_decode(z)
        return A_pred, z,z1,z2

    def dot_product_decode(self, Z):
        x = torch.matmul(Z, Z.t())
        A_pred = torch.sigmoid(x-1/x)
        return A_pred

    def _decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


def GetPrearr(x, num_cluster):
    matrix = np.zeros((len(x), num_cluster))
    for i in range(len(x)):
        for j in range(num_cluster):
            matrix[i][j] = 0
            matrix[i][x[i]] = 1
    return matrix


def Modula(array, cluster):
    m = sum(sum(array)) / 2
    k1 = np.sum(array, axis=1)
    k2 = k1.reshape(k1.shape[0], 1)
    k1k2 = k1 * k2
    Eij = k1k2 / (2 * m)
    B = array - Eij
    node_cluster = np.dot(cluster, np.transpose(cluster))
    results = np.dot(B, node_cluster)
    sum_results = np.trace(results)
    modul = sum_results / (2 * m)
    return modul
