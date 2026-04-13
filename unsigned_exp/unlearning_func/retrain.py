import time
import torch
from train import train_model
from models import get_model


class Retrain:
    """Retrain method - retrain model from scratch without data to be unlearned"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
    
    def unlearn(self, data_processor, original_model, unlearn_nodes=None, unlearn_edges=None):
        """
        Execute retraining
        
        Args:
            data_processor: Data processor
            original_model: Original trained model
            unlearn_nodes: List of nodes to unlearn
            unlearn_edges: List of edges to unlearn
            
        Returns:
            unlearned_model: Retrained model
            unlearn_time: Unlearning time
        """
        start_time = time.time()
        
        data = data_processor.data
        original_train_mask = data_processor.train_mask
        val_mask = data_processor.val_mask
        test_mask = data_processor.test_mask
        
        if unlearn_nodes is not None:
            unlearn_data = data
            unlearn_train_mask = original_train_mask.clone()
            unlearn_train_mask[unlearn_nodes] = False
            
        elif unlearn_edges is not None:
            unlearn_data, unlearn_train_mask = data_processor.create_unlearn_data(
                unlearn_edges=unlearn_edges
            )
        else:
            raise ValueError("Either unlearn_nodes or unlearn_edges must be provided")
        
        num_features = unlearn_data.x.shape[1]
        num_classes = data_processor.num_classes[self.args.dataset]
        
        unlearned_model = get_model(
            self.args.model, 
            num_features, 
            num_classes, 
            self.args
        )
        
        results = train_model(
            unlearned_model, 
            unlearn_data, 
            unlearn_train_mask, 
            val_mask, 
            test_mask, 
            self.args, 
            self.device, 
            verbose=False
        )
        
        unlearn_time = time.time() - start_time
        
        return results['trainer'].model, unlearn_time
    
    def node_unlearn(self, data_processor, original_model, unlearn_nodes):
        """Node unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_nodes=unlearn_nodes)
    
    def edge_unlearn(self, data_processor, original_model, unlearn_edges):
        """Edge unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_edges=unlearn_edges)