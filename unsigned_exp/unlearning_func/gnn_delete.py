import time
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np


class GNNDelete:
    """GNNDelete method - graph unlearning based on deletion operators"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.epochs = args.gnndelete_epochs
        self.lr = args.gnndelete_lr
        self.alpha = args.gnndelete_alpha
        self.loss_type = args.gnndelete_loss_type
    
    def compute_locality_loss(self, model, data, unlearn_nodes, remaining_nodes):
        """Compute locality loss - ensure neighbor embeddings of unlearn nodes remain stable"""
        model.eval()
        
        with torch.no_grad():
            original_embeddings = model.get_embeddings(data.x, data.edge_index)
        
        model.train()
        current_embeddings = model.get_embeddings(data.x, data.edge_index)
        
        edge_index = data.edge_index
        neighbor_nodes = set()
        
        for unlearn_node in unlearn_nodes:
            neighbors = edge_index[1][edge_index[0] == unlearn_node]
            neighbor_nodes.update(neighbors.cpu().numpy())
        
        neighbor_nodes = list(neighbor_nodes - set(unlearn_nodes.cpu().numpy()))
        
        if len(neighbor_nodes) == 0:
            return torch.tensor(0.0).to(self.device)
        
        neighbor_indices = torch.tensor(neighbor_nodes, dtype=torch.long).to(self.device)
        
        if self.loss_type == 'mse':
            locality_loss = F.mse_loss(
                current_embeddings[neighbor_indices],
                original_embeddings[neighbor_indices].detach()
            )
        elif self.loss_type == 'cosine':
            cosine_sim = F.cosine_similarity(
                current_embeddings[neighbor_indices],
                original_embeddings[neighbor_indices].detach(),
                dim=1
            )
            locality_loss = 1 - cosine_sim.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return locality_loss
    
    def compute_randomness_loss(self, model, data, unlearn_nodes):
        """Compute randomness loss - make unlearn nodes' outputs random"""
        model.train()
        out = model(data.x, data.edge_index)
        unlearn_outputs = out[unlearn_nodes]
        
        num_classes = out.shape[1]
        uniform_target = torch.ones_like(unlearn_outputs) / num_classes
        
        if self.loss_type == 'kld':
            log_prob = F.log_softmax(unlearn_outputs, dim=1)
            randomness_loss = F.kl_div(log_prob, uniform_target, reduction='batchmean')
        elif self.loss_type == 'mse':
            prob = F.softmax(unlearn_outputs, dim=1)
            randomness_loss = F.mse_loss(prob, uniform_target)
        elif self.loss_type == 'cosine':
            prob = F.softmax(unlearn_outputs, dim=1)
            cosine_sim = F.cosine_similarity(prob, uniform_target, dim=1)
            randomness_loss = cosine_sim.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return randomness_loss
    
    def unlearn(self, data_processor, original_model, unlearn_nodes=None, unlearn_edges=None):
        """
        Execute GNNDelete unlearning
        
        Args:
            data_processor: Data processor
            original_model: Original trained model
            unlearn_nodes: Nodes to unlearn
            unlearn_edges: Edges to unlearn
            
        Returns:
            unlearned_model: Unlearned model
            unlearn_time: Unlearning time
        """
        start_time = time.time()
        
        from models import get_model
        unlearned_model = get_model(
            self.args.model,
            original_model.num_features,
            original_model.num_classes,
            self.args
        ).to(self.device)
        
        unlearned_model.load_state_dict(original_model.state_dict())
        
        data = data_processor.data.to(self.device)
        train_mask = data_processor.train_mask
        
        if unlearn_edges is not None:
            edge_index = data.edge_index
            affected_nodes = set()
            
            for edge in unlearn_edges.t():
                affected_nodes.add(edge[0].item())
                affected_nodes.add(edge[1].item())
            
            unlearn_nodes = torch.tensor(list(affected_nodes), dtype=torch.long).to(self.device)
            data, _ = data_processor.create_unlearn_data(unlearn_edges=unlearn_edges)
            data = data.to(self.device)
        
        if unlearn_nodes is None:
            raise ValueError("unlearn_nodes must be provided for GNNDelete")
        
        remaining_nodes = torch.where(train_mask)[0]
        remaining_nodes = remaining_nodes[~torch.isin(remaining_nodes, unlearn_nodes)]
        
        optimizer = optim.Adam(unlearned_model.parameters(), lr=self.lr)
        
        print(f"Starting GNNDelete unlearning for {len(unlearn_nodes)} nodes...")
        
        for epoch in range(self.epochs):
            unlearned_model.train()
            optimizer.zero_grad()
            
            randomness_loss = self.compute_randomness_loss(unlearned_model, data, unlearn_nodes)
            locality_loss = self.compute_locality_loss(
                unlearned_model, data, unlearn_nodes, remaining_nodes
            )
            
            total_loss = self.alpha * randomness_loss + (1 - self.alpha) * locality_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % (self.epochs // 10) == 0:
                print(f'Epoch {epoch:03d}, Total Loss: {total_loss:.4f}, '
                      f'Randomness: {randomness_loss:.4f}, Locality: {locality_loss:.4f}')
        
        unlearn_time = time.time() - start_time
        print(f"GNNDelete unlearning completed in {unlearn_time:.2f}s")
        
        return unlearned_model, unlearn_time
    
    def node_unlearn(self, data_processor, original_model, unlearn_nodes):
        """Node unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_nodes=unlearn_nodes)
    
    def edge_unlearn(self, data_processor, original_model, unlearn_edges):
        """Edge unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_edges=unlearn_edges)
    
    def evaluate_unlearning_effectiveness(self, model, data, unlearn_nodes, original_predictions):
        """Evaluate unlearning effectiveness"""
        model.eval()
        with torch.no_grad():
            current_predictions = F.softmax(model(data.x, data.edge_index), dim=1)
        
        unlearn_original = original_predictions[unlearn_nodes]
        unlearn_current = current_predictions[unlearn_nodes]
        
        kl_div = F.kl_div(
            F.log_softmax(unlearn_current, dim=1),
            F.softmax(unlearn_original, dim=1),
            reduction='batchmean'
        )
        
        entropy = -torch.sum(unlearn_current * torch.log(unlearn_current + 1e-8), dim=1).mean()
        
        return {
            'kl_divergence': kl_div.item(),
            'entropy': entropy.item(),
            'max_entropy': np.log(unlearn_current.shape[1])
        }