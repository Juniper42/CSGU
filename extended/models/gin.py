import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm
from torch.nn import Sequential, Linear, ReLU, Dropout


class GIN(nn.Module):
    """Graph Isomorphism Network"""
    
    def __init__(self, num_features, num_classes, hidden_dim=16, num_layers=2, dropout=0.5):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer case
            mlp = Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, num_classes)
            )
            self.convs.append(GINConv(mlp))
            self.classifier = nn.Identity()
        else:
            # Multi-layer case
            # Input layer
            mlp1 = Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp1))
            self.bns.append(BatchNorm(hidden_dim))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                mlp_hidden = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp_hidden))
                self.bns.append(BatchNorm(hidden_dim))
            
            # Final classifier
            self.classifier = Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Dropout(dropout),
                Linear(hidden_dim, num_classes)
            )
    
    def forward(self, x, edge_index, edge_weight=None):
        # Note: GIN doesn't typically use edge_weight, but we keep the interface consistent
        
        if self.num_layers == 1:
            # Single layer case
            x = self.convs[0](x, edge_index)
            return F.log_softmax(x, dim=1)
        
        # Multi-layer case
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """获取节点嵌入（不包含最后的分类层）"""
        if self.num_layers == 1:
            # Single layer case - return features before final linear layer
            x = self.convs[0](x, edge_index)
            # Extract features before the last linear layer
            if hasattr(self.convs[0], 'nn') and len(self.convs[0].nn) >= 3:
                # Get intermediate representation
                mlp = self.convs[0].nn
                for layer in mlp[:-1]:  # All but last layer
                    if hasattr(layer, '__call__'):
                        continue  # Skip if it's already processed
                # Return the hidden representation
                return x[:, :self.hidden_dim] if x.shape[1] > self.hidden_dim else x
            return x
        
        # Multi-layer case
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self, 'bns'):
            for bn in self.bns:
                bn.reset_parameters()
        if hasattr(self.classifier, 'reset_parameters'):
            self.classifier.reset_parameters()
        elif hasattr(self.classifier, 'children'):
            for layer in self.classifier.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()