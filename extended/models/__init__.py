from .gcn import GCN
from .gat import GAT
from .gin import GIN

import torch


def get_model(model_name, num_features, num_classes, args):
    """根据模型名称返回相应的模型实例"""
    
    if model_name.upper() == 'GCN':
        return GCN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    elif model_name.upper() == 'GAT':
        return GAT(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=8 if args.hidden_dim >= 8 else 1
        )
    
    elif model_name.upper() == 'GIN':
        return GIN(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


class ModelWrapper:
    """模型包装器，提供统一的训练和评估接口"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def train_step(self, data, optimizer, criterion, train_mask):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out[mask].max(1)[1]
            correct = pred.eq(data.y[mask]).double()
            accuracy = correct.sum() / len(correct)
        return accuracy.item()
    
    def get_predictions(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            return torch.softmax(out, dim=1)
    
    def get_embeddings(self, data):
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(data.x, data.edge_index)
        return embeddings
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


__all__ = ['GCN', 'GAT', 'GIN', 'get_model', 'ModelWrapper']