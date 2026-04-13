import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, num_features, num_classes, hidden_dim=16, num_layers=2, 
                 dropout=0.5, heads=8, output_heads=1):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_features = num_features
        self.num_classes = num_classes
        self.heads = heads
        self.output_heads = output_heads
        
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            # Single layer case
            self.convs.append(
                GATConv(num_features, num_classes, heads=output_heads, 
                       concat=False, dropout=dropout)
            )
        else:
            # Multi-layer case
            # Input layer
            self.convs.append(
                GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
            )
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
                )
            
            # Output layer
            self.convs.append(
                GATConv(hidden_dim * heads, num_classes, heads=output_heads, 
                       concat=False, dropout=dropout)
            )
    
    def forward(self, x, edge_index, edge_weight=None):
        # Note: GAT doesn't typically use edge_weight, but we keep the interface consistent
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """获取节点嵌入（不包含最后的分类层）"""
        if self.num_layers == 1:
            # Single layer case - return input features
            return x
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()