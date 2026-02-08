import time
import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from torch_geometric.utils import degree


class CSGU:
    """
    Certified Signed Graph Unlearning (CSGU) method - adapted for homogeneous graphs
    Modified: replaced Sociological Influence Quantification with degree-based influence weight calculation
    """
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.alpha = args.csgu_alpha
        self.expansion_depth = args.csgu_expansion_depth
        self.cg_iterations = args.csgu_cg_iterations
        self.damping = args.csgu_damping
        self.hessian_scale = args.csgu_hessian_scale
        self.update_scale = args.csgu_update_scale
        self.epsilon = args.csgu_epsilon
        self.delta = args.csgu_delta
        self.clip_threshold = args.csgu_clip_threshold
    
    def compute_degree_based_influence_weights(self, data, nodes):
        """
        Compute degree-based influence weights
        
        Args:
            data: Graph data
            nodes: Node list
            
        Returns:
            influence_weights: Influence weight vector
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        node_degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        target_degrees = node_degrees[nodes]
        
        max_degree = node_degrees.max()
        min_degree = node_degrees.min()
        
        if max_degree > min_degree:
            influence_weights = 0.1 + 0.9 * (target_degrees - min_degree) / (max_degree - min_degree)
        else:
            influence_weights = torch.ones_like(target_degrees)
        
        if self.expansion_depth > 0:
            influence_weights = self.expand_neighborhood_influence(
                data, nodes, influence_weights
            )
        
        return influence_weights
    
    def expand_neighborhood_influence(self, data, nodes, base_weights):
        """
        Expand neighborhood influence calculation
        
        Args:
            data: Graph data
            nodes: Target nodes
            base_weights: Base weights
            
        Returns:
            expanded_weights: Expanded weights
        """
        edge_index = data.edge_index
        expanded_weights = base_weights.clone()
        
        current_nodes = set(nodes.cpu().numpy())
        
        for depth in range(self.expansion_depth):
            new_nodes = set()
            
            for node in current_nodes:
                neighbors = edge_index[1][edge_index[0] == node]
                new_nodes.update(neighbors.cpu().numpy())
            
            neighbor_nodes = list(new_nodes - current_nodes)
            
            if neighbor_nodes:
                neighbor_indices = torch.tensor(neighbor_nodes, dtype=torch.long).to(self.device)
                neighbor_degrees = degree(edge_index[0], num_nodes=data.num_nodes)[neighbor_indices]
                
                decay_factor = 0.5 ** (depth + 1)
                neighbor_contribution = decay_factor * neighbor_degrees.float()
                
                for i, node_idx in enumerate(nodes):
                    expanded_weights[i] += neighbor_contribution.sum() * 0.1
            
            current_nodes.update(new_nodes)
        
        return expanded_weights
    
    def compute_weighted_gradients(self, model, data, nodes, influence_weights, criterion):
        """Compute weighted gradients"""
        model.train()
        
        output = model(data.x, data.edge_index)
        node_losses = F.nll_loss(output[nodes], data.y[nodes], reduction='none')
        weighted_losses = node_losses * influence_weights
        total_loss = weighted_losses.sum()
        
        gradients = grad(total_loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.view(-1) for g in gradients])
        
        return grad_vector
    
    def compute_hessian_vector_product(self, model, data, train_mask, grad_vector, criterion):
        """Compute Hessian-Vector product"""
        model.train()
        
        output = model(data.x, data.edge_index)
        loss = criterion(output[train_mask], data.y[train_mask])
        
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        loss += self.damping * l2_reg
        
        train_gradients = grad(loss, model.parameters(), create_graph=True)
        
        grad_grad_dot = 0
        for g1, g2 in zip(train_gradients, self._vector_to_parameters(grad_vector, model)):
            grad_grad_dot += torch.sum(g1 * g2)
        
        hvp_gradients = grad(grad_grad_dot, model.parameters(), retain_graph=True)
        hvp_vector = torch.cat([g.view(-1) if g is not None else torch.zeros_like(p.view(-1)) 
                               for g, p in zip(hvp_gradients, model.parameters())])
        
        return hvp_vector * self.hessian_scale
    
    def _vector_to_parameters(self, vector, model):
        """Convert vector back to parameter shapes"""
        params = []
        pointer = 0
        
        for param in model.parameters():
            num_params = param.numel()
            params.append(vector[pointer:pointer + num_params].view(param.shape))
            pointer += num_params
        
        return params
    
    def conjugate_gradient_solve(self, model, data, train_mask, grad_vector, criterion):
        """Solve linear system using conjugate gradient method"""
        x = torch.zeros_like(grad_vector)
        r = grad_vector.clone()
        p = r.clone()
        
        for i in range(self.cg_iterations):
            Ap = self.compute_hessian_vector_product(model, data, train_mask, p, criterion)
            Ap += self.damping * p
            
            r_dot_r = torch.dot(r, r)
            alpha = r_dot_r / torch.dot(p, Ap)
            
            x += alpha * p
            r_new = r - alpha * Ap
            
            if torch.norm(r_new) < 1e-6:
                break
            
            beta = torch.dot(r_new, r_new) / r_dot_r
            p = r_new + beta * p
            r = r_new
        
        return x
    
    def add_differential_privacy_noise(self, vector):
        """Add differential privacy noise"""
        sensitivity = self.clip_threshold
        noise_std = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        noise = torch.randn_like(vector) * noise_std
        return vector + noise
    
    def gradient_clipping(self, vector):
        """Gradient clipping"""
        norm = torch.norm(vector)
        if norm > self.clip_threshold:
            vector = vector * self.clip_threshold / norm
        return vector
    
    def unlearn(self, data_processor, original_model, unlearn_nodes=None, unlearn_edges=None):
        """
        Execute CSGU unlearning
        
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
            raise ValueError("unlearn_nodes must be provided for CSGU")
        
        print(f"Starting CSGU unlearning for {len(unlearn_nodes)} nodes...")
        
        print("Step 1: Computing degree-based influence weights...")
        influence_weights = self.compute_degree_based_influence_weights(data, unlearn_nodes)
        
        print("Step 2: Computing weighted gradients...")
        weighted_gradients = self.compute_weighted_gradients(
            unlearned_model, data, unlearn_nodes, influence_weights, criterion
        )
        
        print("Step 3: Solving for parameter updates...")
        parameter_updates = self.conjugate_gradient_solve(
            unlearned_model, data, train_mask, weighted_gradients, criterion
        )
        
        print("Step 4: Applying gradient clipping...")
        parameter_updates = self.gradient_clipping(parameter_updates)
        
        print("Step 5: Adding differential privacy noise...")
        parameter_updates = self.add_differential_privacy_noise(parameter_updates)
        
        print("Step 6: Applying parameter updates...")
        param_updates = self._vector_to_parameters(parameter_updates, unlearned_model)
        
        with torch.no_grad():
            for param, update in zip(unlearned_model.parameters(), param_updates):
                param.data -= self.update_scale * update
        
        unlearn_time = time.time() - start_time
        print(f"CSGU unlearning completed in {unlearn_time:.2f}s")
        
        certification_bound = self.compute_certification_bound(
            influence_weights, parameter_updates
        )
        
        return {
            'model': unlearned_model,
            'unlearn_time': unlearn_time,
            'certification_bound': certification_bound,
            'influence_weights': influence_weights,
            'update_magnitude': torch.norm(parameter_updates).item()
        }
    
    def compute_certification_bound(self, influence_weights, parameter_updates):
        """Compute certification bound"""
        weight_variance = torch.var(influence_weights)
        update_norm = torch.norm(parameter_updates)
        
        bound = (weight_variance + update_norm) * self.epsilon / (2 * np.log(1 / self.delta))
        
        return bound.item()
    
    def node_unlearn(self, data_processor, original_model, unlearn_nodes):
        """Node unlearning"""
        result = self.unlearn(data_processor, original_model, unlearn_nodes=unlearn_nodes)
        return result['model'], result['unlearn_time']
    
    def edge_unlearn(self, data_processor, original_model, unlearn_edges):
        """Edge unlearning"""
        result = self.unlearn(data_processor, original_model, unlearn_edges=unlearn_edges)
        return result['model'], result['unlearn_time']
    
    def evaluate_unlearning_quality(self, original_model, unlearned_model, data, unlearn_nodes):
        """Evaluate unlearning quality"""
        original_model.eval()
        unlearned_model.eval()
        
        with torch.no_grad():
            original_out = F.softmax(original_model(data.x, data.edge_index), dim=1)
            unlearned_out = F.softmax(unlearned_model(data.x, data.edge_index), dim=1)
            
            unlearn_original = original_out[unlearn_nodes]
            unlearn_new = unlearned_out[unlearn_nodes]
            
            kl_divergence = F.kl_div(
                torch.log(unlearn_new + 1e-8),
                unlearn_original,
                reduction='batchmean'
            )
            
            entropy = -torch.sum(unlearn_new * torch.log(unlearn_new + 1e-8), dim=1).mean()
            
            original_confidence = torch.max(unlearn_original, dim=1)[0].mean()
            new_confidence = torch.max(unlearn_new, dim=1)[0].mean()
            confidence_drop = original_confidence - new_confidence
        
        return {
            'kl_divergence': kl_divergence.item(),
            'entropy': entropy.item(),
            'confidence_drop': confidence_drop.item(),
            'original_confidence': original_confidence.item(),
            'new_confidence': new_confidence.item()
        }