import time
import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np


class GIF:
    """Graph Influence Function (GIF) method - graph unlearning based on influence functions"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.iteration = args.gif_iteration
        self.damp = args.gif_damp
        self.scale = args.gif_scale
    
    def compute_gradients(self, model, data, nodes, criterion):
        """Compute gradients for specified nodes"""
        model.train()
        
        output = model(data.x, data.edge_index)
        loss = criterion(output[nodes], data.y[nodes])
        
        gradients = grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.view(-1) for g in gradients])
        
        return grad_vector
    
    def hvp(self, model, data, train_mask, grad_vector, criterion):
        """Compute Hessian-Vector product (HVP)"""
        model.train()
        
        output = model(data.x, data.edge_index)
        loss = criterion(output[train_mask], data.y[train_mask])
        train_gradients = grad(loss, model.parameters(), create_graph=True)
        
        grad_grad_dot = 0
        for g1, g2 in zip(train_gradients, self._vector_to_parameters(grad_vector, model)):
            grad_grad_dot += torch.sum(g1 * g2)
        
        hvp_gradients = grad(grad_grad_dot, model.parameters(), retain_graph=True)
        hvp_vector = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p.view(-1)) 
                               for g, p in zip(hvp_gradients, model.parameters())])
        
        return hvp_vector
    
    def _vector_to_parameters(self, vector, model):
        """Convert vector back to parameter shapes"""
        params = []
        pointer = 0
        
        for param in model.parameters():
            num_params = param.numel()
            params.append(vector[pointer:pointer + num_params].view(param.shape))
            pointer += num_params
        
        return params
    
    def conjugate_gradient(self, model, data, train_mask, grad_vector, criterion):
        """Solve H^{-1} * grad_vector using conjugate gradient method"""
        x = torch.zeros_like(grad_vector)
        r = grad_vector.clone()
        p = r.clone()
        
        for i in range(self.iteration):
            Ap = self.hvp(model, data, train_mask, p, criterion)
            Ap += self.damp * p
            
            alpha = torch.dot(r, r) / torch.dot(p, Ap)
            x += alpha * p
            
            r_new = r - alpha * Ap
            
            if torch.norm(r_new) < 1e-6:
                break
            
            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            p = r_new + beta * p
            r = r_new
        
        return x
    
    def compute_influence(self, model, data, train_mask, unlearn_nodes, criterion):
        """Compute influence of unlearn nodes on model parameters"""
        print("Computing gradients for unlearn nodes...")
        
        unlearn_gradients = self.compute_gradients(model, data, unlearn_nodes, criterion)
        
        print("Solving inverse Hessian using conjugate gradient...")
        
        influence_vector = self.conjugate_gradient(
            model, data, train_mask, unlearn_gradients, criterion
        )
        
        influence_vector *= self.scale
        
        return influence_vector
    
    def apply_influence_update(self, model, influence_vector):
        """Apply influence update to model parameters"""
        param_updates = self._vector_to_parameters(influence_vector, model)
        
        with torch.no_grad():
            for param, update in zip(model.parameters(), param_updates):
                param.data -= update
    
    def unlearn(self, data_processor, original_model, unlearn_nodes=None, unlearn_edges=None):
        """
        Execute GIF unlearning
        
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
        criterion = F.nll_loss
        
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
            raise ValueError("unlearn_nodes must be provided for GIF")
        
        print(f"Starting GIF unlearning for {len(unlearn_nodes)} nodes...")
        
        influence_vector = self.compute_influence(
            unlearned_model, data, train_mask, unlearn_nodes, criterion
        )
        
        print("Applying influence update...")
        
        self.apply_influence_update(unlearned_model, influence_vector)
        
        unlearn_time = time.time() - start_time
        print(f"GIF unlearning completed in {unlearn_time:.2f}s")
        
        return unlearned_model, unlearn_time
    
    def node_unlearn(self, data_processor, original_model, unlearn_nodes):
        """Node unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_nodes=unlearn_nodes)
    
    def edge_unlearn(self, data_processor, original_model, unlearn_edges):
        """Edge unlearning"""
        return self.unlearn(data_processor, original_model, unlearn_edges=unlearn_edges)
    
    def evaluate_influence_magnitude(self, influence_vector):
        """Evaluate influence vector magnitude"""
        return {
            'l1_norm': torch.norm(influence_vector, p=1).item(),
            'l2_norm': torch.norm(influence_vector, p=2).item(),
            'max_abs': torch.max(torch.abs(influence_vector)).item(),
            'mean_abs': torch.mean(torch.abs(influence_vector)).item()
        }