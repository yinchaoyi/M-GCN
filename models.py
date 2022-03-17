import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCNThreeOmics(nn.Module):
    def __init__(self, feature_1_num, feature_2_num, feature_3_num, nfeat1, nfeat2, nfeat3, nhid, nclass, dropout=0.):
        super(GCNThreeOmics, self).__init__()
        self.feature_1_num = feature_1_num
        self.feature_2_num = feature_2_num
        self.feature_3_num = feature_3_num
        self.fc1 = nn.Linear(feature_1_num, nfeat1, bias=True)
        self.fc2 = nn.Linear(feature_2_num, nfeat2, bias=True)
        self.fc3 = nn.Linear(feature_3_num, nfeat3, bias=True)
        self.act_fn = nn.ReLU()
        self.gc1 = GraphConvolution(nfeat1 + nfeat2 + nfeat3, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = self.act_fn(self.fc1(x[:, :self.feature_1_num]))
        x2 = self.act_fn(self.fc2(x[:, self.feature_1_num:self.feature_2_num + self.feature_1_num]))
        x3 = self.act_fn(self.fc3(x[:, self.feature_2_num + self.feature_1_num:]))
        x = torch.cat((x1, x2, x3), 1)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
