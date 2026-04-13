import time
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from models import get_model
from train import Trainer


class GraphEraser:
    """GraphEraser method - shard-based graph unlearning method"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_shards = args.eraser_num_shards
        self.aggregation = args.eraser_aggregation
    
    def create_shards(self, data, train_mask, num_shards):
        """Divide training data into multiple shards"""
        train_indices = torch.where(train_mask)[0]
        
        perm = torch.randperm(len(train_indices))
        shuffled_indices = train_indices[perm]
        
        shard_size = len(shuffled_indices) // num_shards
        shards = []
        
        for i in range(num_shards):
            start_idx = i * shard_size
            if i == num_shards - 1:
                end_idx = len(shuffled_indices)
            else:
                end_idx = (i + 1) * shard_size
            
            shard_indices = shuffled_indices[start_idx:end_idx]
            
            shard_mask = torch.zeros_like(train_mask)
            shard_mask[shard_indices] = True
            
            shards.append({
                'mask': shard_mask,
                'indices': shard_indices
            })
        
        return shards
    
    def train_shard_models(self, data, shards, val_mask, test_mask):
        """Train model for each shard"""
        shard_models = []
        
        num_features = data.x.shape[1]
        num_classes = self.get_num_classes(data)
        
        for i, shard in enumerate(shards):
            print(f"Training shard {i+1}/{len(shards)} with {shard['mask'].sum()} nodes...")
            
            from models import get_model
            model = get_model(self.args.model, num_features, num_classes, self.args)
            
            from train import Trainer
            trainer = Trainer(model, data, self.args, self.device)
            trainer.train(shard['mask'], val_mask, verbose=False)
            
            shard_models.append(trainer.model)
        
        return shard_models
    
    def aggregate_predictions(self, shard_models, data, mask=None):
        """Aggregate predictions from multiple shard models"""
        predictions = []
        
        for model in shard_models:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = F.softmax(out, dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        if self.aggregation == 'mean':
            aggregated = torch.mean(predictions, dim=0)
        elif self.aggregation == 'max':
            aggregated, _ = torch.max(predictions, dim=0)
        elif self.aggregation == 'min':
            aggregated, _ = torch.min(predictions, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        if mask is not None:
            return aggregated[mask]
        return aggregated
    
    def get_num_classes(self, data):
        """Get number of classes"""
        return int(data.y.max().item()) + 1
    
    def unlearn(self, data_processor, original_model, unlearn_nodes=None, unlearn_edges=None):
        """
        Execute GraphEraser unlearning
        
        Args:
            data_processor: Data processor
            original_model: Original model (not needed for GraphEraser, but kept for interface consistency)
            unlearn_nodes: Nodes to unlearn
            unlearn_edges: Edges to unlearn
            
        Returns:
            unlearned_models: List of unlearned models
            unlearn_time: Unlearning time
        """
        start_time = time.time()
        
        data = data_processor.data.to(self.device)
        train_mask = data_processor.train_mask
        val_mask = data_processor.val_mask
        test_mask = data_processor.test_mask
        
        if unlearn_nodes is not None:
            remaining_train_mask = train_mask.clone()
            remaining_train_mask[unlearn_nodes] = False
            unlearn_data = data
            
        elif unlearn_edges is not None:
            unlearn_data, remaining_train_mask = data_processor.create_unlearn_data(
                unlearn_edges=unlearn_edges
            )
            unlearn_data = unlearn_data.to(self.device)
        else:
            raise ValueError("Either unlearn_nodes or unlearn_edges must be provided")
        
        shards = self.create_shards(unlearn_data, remaining_train_mask, self.num_shards)
        shard_models = self.train_shard_models(unlearn_data, shards, val_mask, test_mask)
        
        unlearn_time = time.time() - start_time
        
        return shard_models, unlearn_time
    
    def node_unlearn(self, data_processor, original_model, unlearn_nodes):
        """Node unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_nodes=unlearn_nodes)
    
    def edge_unlearn(self, data_processor, original_model, unlearn_edges):
        """Edge unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_edges=unlearn_edges)
    
    def evaluate_ensemble(self, shard_models, data, mask):
        """Evaluate ensemble model performance"""
        aggregated_pred = self.aggregate_predictions(shard_models, data, mask)
        pred_labels = aggregated_pred.max(1)[1]
        true_labels = data.y[mask]
        
        correct = pred_labels.eq(true_labels).double()
        accuracy = correct.sum() / len(correct)
        
        return accuracy.item()


class GraphEraserWrapper:
    """GraphEraser wrapper that provides interface consistent with single models"""
    
    def __init__(self, shard_models, aggregation='mean'):
        self.shard_models = shard_models
        self.aggregation = aggregation
        self.device = shard_models[0].parameters().__next__().device
    
    def __call__(self, x, edge_index):
        """Forward propagation"""
        predictions = []
        
        for model in self.shard_models:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                predictions.append(out)
        
        predictions = torch.stack(predictions)
        
        if self.aggregation == 'mean':
            return torch.mean(predictions, dim=0)
        elif self.aggregation == 'max':
            return torch.max(predictions, dim=0)[0]
        elif self.aggregation == 'min':
            return torch.min(predictions, dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def eval(self):
        """Set to evaluation mode"""
        for model in self.shard_models:
            model.eval()
    
    def to(self, device):
        """Move to specified device"""
        for i, model in enumerate(self.shard_models):
            self.shard_models[i] = model.to(device)
        self.device = device
        return self