import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm


class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, num_features, num_classes, hidden_dim=16, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        if num_layers > 1:
            self.bns.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, num_classes))
        else:
            # Single layer case
            self.convs[0] = GCNConv(num_features, num_classes)
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if hasattr(self, 'bns') and i < len(self.bns):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """获取节点嵌入（不包含最后的分类层）"""
        if self.num_layers == 1:
            # Single layer case - return features before activation
            return self.convs[0](x, edge_index, edge_weight)
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if hasattr(self, 'bns') and i < len(self.bns):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self, 'bns'):
            for bn in self.bns:
                bn.reset_parameters()