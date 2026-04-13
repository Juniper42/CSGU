import time
import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np


class IDEA:
    """IDEA method - Improving Deletion and Explanation Approximation for Graph Unlearning"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.iteration = args.idea_iteration
        self.damp = args.idea_damp
        self.scale = args.idea_scale
        self.gaussian_std = args.idea_gaussian_std
        self.gaussian_mean = args.idea_gaussian_mean
        self.lipschitz_constant = args.idea_l
        self.strong_convexity = args.idea_lambda
        self.loss_bound = args.idea_c
    
    def add_gaussian_noise(self, tensor):
        """Add Gaussian noise for differential privacy protection"""
        noise = torch.randn_like(tensor) * self.gaussian_std + self.gaussian_mean
        return tensor + noise
    
    def compute_loss_with_regularization(self, model, data, nodes, criterion):
        """Compute loss with regularization"""
        output = model(data.x, data.edge_index)
        base_loss = criterion(output[nodes], data.y[nodes])
        
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        
        total_loss = base_loss + self.strong_convexity * l2_reg
        return total_loss
    
    def compute_gradients(self, model, data, nodes, criterion):
        """Compute gradients for specified nodes with noise"""
        model.train()
        
        loss = self.compute_loss_with_regularization(model, data, nodes, criterion)
        gradients = grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.view(-1) for g in gradients])
        grad_vector = self.add_gaussian_noise(grad_vector)
        
        return grad_vector
    
    def hvp_with_damping(self, model, data, train_mask, grad_vector, criterion):
        """Compute Hessian-Vector product with damping"""
        model.train()
        
        loss = self.compute_loss_with_regularization(model, data, train_mask, criterion)
        train_gradients = grad(loss, model.parameters(), create_graph=True)
        
        grad_grad_dot = 0
        for g1, g2 in zip(train_gradients, self._vector_to_parameters(grad_vector, model)):
            grad_grad_dot += torch.sum(g1 * g2)
        
        hvp_gradients = grad(grad_grad_dot, model.parameters(), retain_graph=True)
        hvp_vector = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p.view(-1)) 
                               for g, p in zip(hvp_gradients, model.parameters())])
        
        hvp_vector += self.damp * grad_vector
        hvp_vector = self.add_gaussian_noise(hvp_vector)
        
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
    
    def conjugate_gradient_with_certification(self, model, data, train_mask, grad_vector, criterion):
        """Solve H^{-1} * grad_vector using conjugate gradient with certification"""
        x = torch.zeros_like(grad_vector)
        r = grad_vector.clone()
        p = r.clone()
        
        residual_norms = []
        
        for i in range(self.iteration):
            Ap = self.hvp_with_damping(model, data, train_mask, p, criterion)
            
            r_dot_r = torch.dot(r, r)
            p_dot_Ap = torch.dot(p, Ap)
            
            if p_dot_Ap.item() < 1e-10:
                break
            
            alpha = r_dot_r / p_dot_Ap
            x += alpha * p
            
            r_new = r - alpha * Ap
            residual_norm = torch.norm(r_new)
            residual_norms.append(residual_norm.item())
            
            if residual_norm < 1e-6:
                break
            
            beta = torch.dot(r_new, r_new) / r_dot_r
            p = r_new + beta * p
            r = r_new
        
        certification_bound = self.compute_certification_bound(residual_norms)
        
        return x, certification_bound
    
    def compute_certification_bound(self, residual_norms):
        """Compute IDEA certification bound"""
        if not residual_norms:
            return float('inf')
        
        final_residual = residual_norms[-1]
        noise_level = self.gaussian_std
        bound = (self.loss_bound * final_residual + noise_level) / self.strong_convexity
        
        return bound
    
    def compute_influence_with_certification(self, model, data, train_mask, unlearn_nodes, criterion):
        """Compute influence function with certification"""
        print("Computing gradients for unlearn nodes with noise...")
        
        unlearn_gradients = self.compute_gradients(model, data, unlearn_nodes, criterion)
        
        print("Solving inverse Hessian with certification...")
        
        influence_vector, cert_bound = self.conjugate_gradient_with_certification(
            model, data, train_mask, unlearn_gradients, criterion
        )
        
        influence_vector *= self.scale
        
        print(f"Certification bound: {cert_bound:.6f}")
        
        return influence_vector, cert_bound
    
    def apply_influence_update_with_clipping(self, model, influence_vector):
        """Apply influence update to model parameters with gradient clipping"""
        param_updates = self._vector_to_parameters(influence_vector, model)
        
        total_norm = torch.norm(influence_vector)
        max_norm = 1.0
        if total_norm > max_norm:
            clip_coeff = max_norm / total_norm
            influence_vector *= clip_coeff
            param_updates = self._vector_to_parameters(influence_vector, model)
            print(f"Clipped gradient norm from {total_norm:.4f} to {max_norm}")
        
        with torch.no_grad():
            for param, update in zip(model.parameters(), param_updates):
                param.data -= update
    
    def unlearn(self, data_processor, original_model, unlearn_nodes=None, unlearn_edges=None):
        """
        Execute IDEA unlearning
        
        Args:
            data_processor: Data processor
            original_model: Original trained model
            unlearn_nodes: Nodes to unlearn
            unlearn_edges: Edges to unlearn
            
        Returns:
            result_dict: Contains unlearned model, time and certification info
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
            raise ValueError("unlearn_nodes must be provided for IDEA")
        
        print(f"Starting IDEA unlearning for {len(unlearn_nodes)} nodes...")
        
        influence_vector, certification_bound = self.compute_influence_with_certification(
            unlearned_model, data, train_mask, unlearn_nodes, criterion
        )
        
        print("Applying influence update with clipping...")
        
        self.apply_influence_update_with_clipping(unlearned_model, influence_vector)
        
        unlearn_time = time.time() - start_time
        print(f"IDEA unlearning completed in {unlearn_time:.2f}s")
        
        return {
            'model': unlearned_model,
            'unlearn_time': unlearn_time,
            'certification_bound': certification_bound,
            'influence_magnitude': torch.norm(influence_vector).item()
        }
    
    def node_unlearn(self, data_processor, original_model, unlearn_nodes):
        """Node unlearning"""
        result = self.unlearn(data_processor, original_model, unlearn_nodes=unlearn_nodes)
        return result['model'], result['unlearn_time']
    
    def edge_unlearn(self, data_processor, original_model, unlearn_edges):
        """Edge unlearning"""
        result = self.unlearn(data_processor, original_model, unlearn_edges=unlearn_edges)
        return result['model'], result['unlearn_time']
    
    def evaluate_privacy_guarantee(self, certification_bound, epsilon=1.0):
        """Evaluate privacy guarantee"""
        privacy_cost = certification_bound / epsilon
        
        return {
            'certification_bound': certification_bound,
            'privacy_cost': privacy_cost,
            'privacy_satisfied': privacy_cost <= 1.0
        }