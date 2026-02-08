import copy
import time
import math

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch_sparse import spspmm
from torch_scatter import scatter_add
import numpy as np


class CSGU:
    """
    Certified Signed Graph Unlearning (CSGU) - A certified approach based on social theories.
    This method combines Balance Theory and Status Theory through three core components:
    1. Triangle-based Influence Region Analysis
    2. Sociological Influence Quantification 
    3. Weighted Influence Function
    """

    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device

        # CSGU-specific parameters from args
        self.alpha = getattr(args, 'csgu_alpha', 0.5)  # Balance theory vs Status theory weight
        self.triangle_expansion_depth = getattr(args, 'csgu_expansion_depth', 1)  # p parameter
        self.cg_iterations = getattr(args, 'csgu_cg_iterations', 20)  # T parameter
        self.damping = getattr(args, 'csgu_damping', 0.1)  # λ parameter
        self.hessian_scale = getattr(args, 'csgu_hessian_scale', 1.0)  # σ parameter
        self.update_scale = getattr(args, 'csgu_update_scale', 0.1)  # η parameter
        self.epsilon = getattr(args, 'csgu_epsilon', 1.0)  # Privacy budget
        self.delta = getattr(args, 'csgu_delta', 1e-5)  # Failure probability
        self.clip_threshold = getattr(args, 'csgu_clip_threshold', 1.0)  # Gradient clipping threshold
        
        # Pre-process graph data for efficient computations
        self._preprocess_graph_data()

    def _preprocess_graph_data(self):
        """Pre-processes graph data to create adjacency structures for efficient lookups."""
        self.train_edges = self.data['train']['edges'].to(self.device)
        self.train_labels = self.data['train']['label'].to(self.device)
        self.num_nodes = self.train_edges.max().item() + 1
        
        # Create edge-to-index mapping for efficient triangle detection
        edge_to_idx = {}
        for idx, (u, v) in enumerate(self.train_edges.cpu().numpy()):
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx  # Treat as undirected for triangle detection
        self.edge_to_idx = edge_to_idx
        
        # Create adjacency lists for triangle finding
        self.adj_list = [set() for _ in range(self.num_nodes)]
        for u, v in self.train_edges.cpu().numpy():
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
        
        # Convert signs from {0,1} to {-1,+1} for social theories
        self.signs = self.train_labels * 2 - 1
        self.node_degrees = torch.bincount(self.train_edges.flatten(), minlength=self.num_nodes).to(self.device)

        # Sparse matrices for vectorized computations
        edge_index = self.train_edges.t().contiguous()
        self.A_sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=self.device),
            (self.num_nodes, self.num_nodes)
        ).coalesce()
        self.S_sparse = torch.sparse_coo_tensor(
            edge_index,
            self.signs.float(),
            (self.num_nodes, self.num_nodes)
        ).coalesce()

    def _determine_target_edges(self, unlearned_data):
        """Determines target edges based on unlearning task."""
        if self.args.unlearning_task == 'edge':
            return unlearned_data['unlearned_edges'].to(self.device)
        elif self.args.unlearning_task == 'node':
            unlearned_nodes = unlearned_data['unlearned_nodes'].to(self.device)
            mask = torch.isin(self.train_edges, unlearned_nodes).any(dim=1)
            return self.train_edges[mask]
        elif self.args.unlearning_task == 'node_feature':
            # For node feature unlearning, return empty tensor as no specific edges need to be targeted
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 2)
        else:
            raise ValueError(f"Unknown unlearning task: {self.args.unlearning_task}")

    def _triangle_condition(self, edge1, edge2):
        """
        Checks if two edges form a triangle (share exactly one node and the third edge exists).
        edge1: (u1, v1), edge2: (u2, v2)
        Returns True if they form a triangle, False otherwise.
        """
        u1, v1 = edge1
        u2, v2 = edge2
        
        # Find shared node
        shared_nodes = set([u1, v1]) & set([u2, v2])
        if len(shared_nodes) != 1:
            return False
        
        shared = shared_nodes.pop()
        
        # Find the third node from each edge
        if u1 == shared:
            third1 = v1
        else:
            third1 = u1
            
        if u2 == shared:
            third2 = v2
        else:
            third2 = u2
        
        # Check if the third edge exists
        return (third1, third2) in self.edge_to_idx or (third2, third1) in self.edge_to_idx

    def _triangle_expansion_step(self, current_influence_edges):
        """
        Performs one step of triangle-based expansion.
        Returns new edges that form triangles with current influence edges.
        """
        if current_influence_edges.size(0) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 2)
        
        new_edges_set = set()
        
        # Enumerate triangles via common neighbors of endpoints
        for x, y in current_influence_edges.cpu().numpy():
            if x >= len(self.adj_list) or y >= len(self.adj_list):
                continue
            common_neighbors = self.adj_list[x] & self.adj_list[y]
            if not common_neighbors:
                continue
            for z in common_neighbors:
                # Candidate edges that share exactly one node with (x,y)
                if (x, z) in self.edge_to_idx:
                    new_edges_set.add((x, z))
                if (y, z) in self.edge_to_idx:
                    new_edges_set.add((y, z))
        
        if not new_edges_set:
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 2)
        
        return torch.tensor(list(new_edges_set), dtype=torch.long, device=self.device)

    def _triangle_based_influence_region(self, target_edges):
        """
        Computes the triangle-based influence region using p-hop triangle expansion.
        
        Algorithm:
        D_inf^(0) = E_target
        D_inf^(k) = D_inf^(k-1) ∪ {(u,v,s) ∈ D \ D_inf^(k-1) : ∃(x,y,s') ∈ D_inf^(k-1), Triangle((u,v), (x,y))}
        """
        if target_edges.size(0) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # Initialize with target edges
        current_influence = target_edges.clone()
        all_influence_edges = set()
        
        # Add target edges to influence set
        for edge in target_edges.cpu().numpy():
            all_influence_edges.add((edge[0], edge[1]))
        
        # Perform triangle expansion for p steps
        for step in range(self.triangle_expansion_depth):
            new_edges = self._triangle_expansion_step(current_influence)
            
            if new_edges.size(0) == 0:
                break
            
            # Filter out edges that are already in influence region
            filtered_new_edges = []
            for edge in new_edges.cpu().numpy():
                if (edge[0], edge[1]) not in all_influence_edges:
                    filtered_new_edges.append(edge)
                    all_influence_edges.add((edge[0], edge[1]))
            
            if len(filtered_new_edges) == 0:
                break
                
            filtered_new_edges_tensor = torch.tensor(filtered_new_edges, dtype=torch.long, device=self.device)
            current_influence = torch.cat([current_influence, filtered_new_edges_tensor], dim=0)
        
        # Convert edge set back to tensor
        influence_edges_list = list(all_influence_edges)
        if len(influence_edges_list) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        influence_edges = torch.tensor(influence_edges_list, dtype=torch.long, device=self.device)
        
        # Find corresponding indices in training data
        influence_indices = []
        for u, v in influence_edges.cpu().numpy():
            idx = self.edge_to_idx.get((u, v))
            if idx is None:
                idx = self.edge_to_idx.get((v, u))
            if idx is not None:
                influence_indices.append(idx)
        
        return torch.tensor(influence_indices, dtype=torch.long, device=self.device)

    def _compute_balance_centrality(self, influence_nodes):
        """
        Computes balance centrality for given nodes using GPU-optimized approach.
        """
        if influence_nodes.numel() == 0:
            return torch.zeros(0, device=self.device)

        with torch.no_grad():
            A2 = torch.sparse.mm(self.A_sparse, self.A_sparse).coalesce()
            A3_diag = (A2 * self.A_sparse).sum(dim=1).to_dense()

            S2 = torch.sparse.mm(self.S_sparse, self.S_sparse).coalesce()
            S3_diag = (S2 * self.S_sparse).sum(dim=1).to_dense()

        num_triangles = A3_diag * 0.5
        num_balanced_triangles = (A3_diag + S3_diag) * 0.25
        
        balance_scores = torch.zeros(self.num_nodes, device=self.device)
        mask = num_triangles > 0
        balance_scores[mask] = num_balanced_triangles[mask] / num_triangles[mask]
        
        return balance_scores[influence_nodes]
    
    def _compute_status_centrality(self, influence_nodes):
        """
        Computes status centrality for given nodes.
        
        SC(v) = (1/√|N(v)|) * Σ_{u ∈ N(v)} S(u,v) * σ(deg(u)) if |N(v)| > 0, else 0
        where σ(x) = 1/(1 + e^(-x)) is sigmoid function
        """
        if influence_nodes.numel() == 0:
            return torch.zeros(0, device=self.device)

        # Vectorized computation of Σ_{u ∈ N(v)} S(u,v) * σ(deg(u))
        neighbor_degrees = self.node_degrees[self.train_edges[:, 1]]
        sigmoid_degrees = torch.sigmoid(neighbor_degrees.float())
        weighted_signs = self.signs * sigmoid_degrees
        
        # Sum weighted signs for each node
        status_sum = scatter_add(weighted_signs, self.train_edges[:, 0], dim=0, dim_size=self.num_nodes)
        
        # Normalize by √|N(v)|
        sqrt_degrees = torch.sqrt(self.node_degrees.float())
        # Avoid division by zero for isolated nodes
        sqrt_degrees[sqrt_degrees == 0] = 1.0
        
        status_scores = status_sum / sqrt_degrees
        
        return status_scores[influence_nodes]

    def _min_max_normalize(self, scores):
        """Min-max normalization to scale scores to [0,1] range."""
        if scores.numel() <= 1:
            return torch.zeros_like(scores, dtype=torch.float32)
        
        score_min = scores.min()
        score_max = scores.max()
        if score_max == score_min:
            return torch.zeros_like(scores, dtype=torch.float32)
        
        return (scores - score_min) / (score_max - score_min)

    def _compute_unified_centrality_and_influence(self, influence_nodes):
        """
        Computes unified centrality and node influence distribution.
        
        Steps:
        1. Normalize balance and status centralities to [0,1]
        2. Combine: UC(v) = α * N(BC(v)) + (1-α) * N(|SC(v)|)
        3. Apply softmax: I(v) = exp(UC(v)) / Σ exp(UC(u))
        """
        if influence_nodes.numel() == 0:
            return {}
        
        # Step 1: Compute balance and status centralities
        balance_centrality = self._compute_balance_centrality(influence_nodes)
        status_centrality = self._compute_status_centrality(influence_nodes)
        
        # Step 2: Normalize both centralities to [0,1] range
        balance_normalized = self._min_max_normalize(balance_centrality)
        status_normalized = self._min_max_normalize(torch.abs(status_centrality))  # Use absolute value as in formula
        
        # Step 3: Combine using α parameter
        unified_centrality = (self.alpha * balance_normalized + 
                            (1 - self.alpha) * status_normalized)
        
        # Step 4: Apply softmax to get probability distribution
        if unified_centrality.numel() == 0:
            return {}
        
        # Use torch.softmax for numerical stability
        influence_distribution = torch.softmax(unified_centrality, dim=0)
        
        # Step 5: Convert to dictionary mapping node_id -> influence_score
        node_influence = {}
        for i, node in enumerate(influence_nodes):
            node_influence[node.item()] = influence_distribution[i].item()
        
        return node_influence

    def _compute_weighted_bce_loss(self, model, edges, labels, node_weights):
        """
        GPU-optimized weighted BCE loss using vectorized operations.
        
        L_w = Σ_{(u,v,s) ∈ D'} -w_uv * [s' log σ(ŝ_uv) + (1-s') log(1-σ(ŝ_uv))]
        where w_uv = (I(u) + I(v)) / 2
        """
        if edges.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Get model predictions - vectorized
        node_embeddings = model()
        src_emb = node_embeddings[edges[:, 0]]
        dst_emb = node_embeddings[edges[:, 1]]
        logits = (src_emb * dst_emb).sum(dim=1)
        
        # Vectorized edge weight computation
        node_weight_tensor = torch.zeros(self.num_nodes, device=self.device, dtype=logits.dtype)
        if len(node_weights) > 0:
            idxs = torch.as_tensor(list(node_weights.keys()), device=self.device, dtype=torch.long)
            vals = torch.as_tensor(list(node_weights.values()), device=self.device, dtype=logits.dtype)
            node_weight_tensor[idxs] = vals
        edge_weights = (node_weight_tensor[edges[:, 0]] + node_weight_tensor[edges[:, 1]]) * 0.5
        
        # Weighted BCE loss - all GPU operations
        weighted_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), weight=edge_weights, reduction='sum'
        )
        
        return weighted_loss

    def _compute_gradient_difference(self, model, influence_indices, unlearned_data, node_weights):
        """
        Computes gradient difference: g = ∇L_w(Z) - ∇L_w(Z \ D_u)
        """
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        # Get influence region edges and labels
        if influence_indices.size(0) == 0:
            return [torch.zeros_like(p) for p in model_params]
        
        influence_edges = self.train_edges[influence_indices]
        influence_labels = self.train_labels[influence_indices]
        
        # Compute full influence region loss
        full_loss = self._compute_weighted_bce_loss(model, influence_edges, influence_labels, node_weights)
        
        # Identify unlearned edges within influence region
        if self.args.unlearning_task == 'edge':
            unlearned_edges = unlearned_data['unlearned_edges'].to(self.device)
            unlearned_labels = unlearned_data['unlearned_labels'].to(self.device)
        elif self.args.unlearning_task == 'node':
            unlearned_nodes = unlearned_data['unlearned_nodes'].to(self.device)
            mask = torch.isin(self.train_edges, unlearned_nodes).any(dim=1)
            unlearned_edges = self.train_edges[mask]
            unlearned_labels = self.train_labels[mask]
        elif self.args.unlearning_task == 'node_feature':
            # For node feature unlearning, no specific edges are unlearned
            unlearned_edges = torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 2)
            unlearned_labels = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Filter unlearned edges that are in influence region using canonical encoding
        def canonical_keys(edges):
            u = torch.min(edges[:, 0], edges[:, 1])
            v = torch.max(edges[:, 0], edges[:, 1])
            return u.to(torch.long) * self.num_nodes + v.to(torch.long)

        inf_keys = canonical_keys(influence_edges)
        unl_keys = canonical_keys(unlearned_edges)
        mask = torch.isin(unl_keys, inf_keys)

        if not mask.any():
            return [torch.zeros_like(p) for p in model_params]

        unlearned_in_influence = unlearned_edges[mask]
        unlearned_labels_in_influence = unlearned_labels[mask]
        
        # Compute unlearned loss within influence region
        unlearned_loss = self._compute_weighted_bce_loss(
            model, unlearned_in_influence, unlearned_labels_in_influence, node_weights
        )
        
        # Compute gradients
        grad_full = grad(full_loss, model_params, retain_graph=True, create_graph=True, allow_unused=True)
        grad_unlearned = grad(unlearned_loss, model_params, retain_graph=True, create_graph=True, allow_unused=True)
        
        grad_full = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_full, model_params)]
        grad_unlearned = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_unlearned, model_params)]
        
        # Gradient difference
        gradient_diff = [gf - gu for gf, gu in zip(grad_full, grad_unlearned)]
        
        return gradient_diff

    def _compute_hessian_vector_product(self, model, v):
        """Computes Hessian-vector product HVP(∇_train, v)."""
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        # Compute full training loss for Hessian computation
        node_embeddings = model()
        src_emb = node_embeddings[self.train_edges[:, 0]]
        dst_emb = node_embeddings[self.train_edges[:, 1]]
        logits = (src_emb * dst_emb).sum(dim=1)
        train_loss = F.binary_cross_entropy_with_logits(logits, self.train_labels.float(), reduction='mean')
        
        # Compute gradient
        train_grad = grad(train_loss, model_params, retain_graph=True, create_graph=True, allow_unused=True)
        train_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(train_grad, model_params)]
        
        # Compute gradient-vector product
        gv_product = sum(
            (g * v_elem.detach()).sum()
            for g, v_elem in zip(train_grad, v)
            if g is not None and v_elem is not None
        )
        
        # Compute Hessian-vector product
        hvp = grad(gv_product, model_params, create_graph=True, allow_unused=True)
        return [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp, model_params)]

    def _certified_conjugate_gradient(self, model, gradient_diff, unlearned_data):
        """
        Implements certified conjugate gradient algorithm (Algorithm 1 from SGU paper).
        """
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        # Step 1: Compute sensitivity (approximation)
        delta_max = 0.0
        for g in gradient_diff:
            if g is not None:
                delta_max = max(delta_max, g.norm().item())
        
        if delta_max == 0:
            return [torch.zeros_like(p) for p in model_params]
        
        # Step 2: Compute noise variance calibration  
        num_unlearned = 1  # Simplified for now
        if self.args.unlearning_task == 'edge':
            num_unlearned = unlearned_data['unlearned_edges'].size(0)
        elif self.args.unlearning_task == 'node':
            num_unlearned = unlearned_data['unlearned_nodes'].size(0)
        elif self.args.unlearning_task == 'node_feature':
            num_unlearned = unlearned_data.get('unlearned_nodes_num', 1)
        
        noise_variance = (2 * delta_max * math.sqrt(2 * math.log(1.25 / self.delta)) / 
                         (num_unlearned * self.epsilon)) ** 2
        
        # Step 3: Initialize h^(0) = g
        h_estimate = gradient_diff
        
        # Step 4: Iterative updates
        for t in range(self.cg_iterations):
            # Compute HVP
            hvp = self._compute_hessian_vector_product(model, h_estimate)
            
            # Update: h^(t) = g + (1-λ)h^(t-1) - (1/σ)HVP(∇_train, h^(t-1))
            with torch.no_grad():
                h_estimate = [
                    g + (1 - self.damping) * h - hvp_elem / self.hessian_scale
                    for g, h, hvp_elem in zip(gradient_diff, h_estimate, hvp)
                ]
            
            # Check for numerical stability
            h_norm = sum(h.norm().item() for h in h_estimate)
            if math.isnan(h_norm) or h_norm > 1e6:
                break
        
        # Step 5: Add certified noise
        for i, h in enumerate(h_estimate):
            if h.numel() > 0:
                noise = torch.normal(0, math.sqrt(noise_variance), h.shape, device=self.device)
                h_estimate[i] = h + noise
        
        return h_estimate

    def _clip_and_update_parameters(self, model, param_changes):
        """Applies gradient clipping and updates model parameters."""
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        with torch.no_grad():
            for p, change in zip(model_params, param_changes):
                if change.numel() > 0:
                    # Apply gradient clipping
                    change_norm = change.norm()
                    if change_norm > self.clip_threshold:
                        change = change * (self.clip_threshold / change_norm)
                    
                    # Update parameter with scaling
                    p.add_(change * self.update_scale)

    def unlearn(self, original_model, unlearned_data):
        """
        Executes the SGU unlearning process.
        """
        unlearned_model = copy.deepcopy(original_model)
        unlearned_model.to(self.device)
        unlearned_model.eval()

        start_time = time.time()

        # Step 1: Triangle-based influence region analysis
        target_edges = self._determine_target_edges(unlearned_data)
        influence_indices = self._triangle_based_influence_region(target_edges)
        
        # Get unique nodes in influence region
        if influence_indices.size(0) > 0:
            influence_edges = self.train_edges[influence_indices]
            influence_nodes = torch.unique(influence_edges.flatten())
        else:
            influence_nodes = torch.tensor([], dtype=torch.long, device=self.device)
        
        print(f"Target edges: {target_edges.size(0)}, Influence region size: {influence_indices.size(0)}")
        print(f"Influence nodes: {influence_nodes.size(0)}")
        print(f"Step 1 (Triangle-based influence region) time: {time.time() - start_time:.4f}s")
        
        if influence_nodes.size(0) == 0:
            print("No influence region found, returning original model")
            unlearning_time = time.time() - start_time
            return unlearning_time, unlearned_model
        
        # Step 2: Sociological influence quantification
        step2_start = time.time()
        node_importance = self._compute_unified_centrality_and_influence(influence_nodes)
        print(f"Step 2 (Sociological influence quantification) time: {time.time() - step2_start:.4f}s")
        
        # Step 3: Weighted influence function
        step3_start = time.time()
        gradient_diff = self._compute_gradient_difference(unlearned_model, influence_indices, unlearned_data, node_importance)
        print(f"Step 3 (Gradient computation) time: {time.time() - step3_start:.4f}s")
        
        # Step 4: Certified conjugate gradient algorithm
        step4_start = time.time()
        param_changes = self._certified_conjugate_gradient(unlearned_model, gradient_diff, unlearned_data)
        print(f"Step 4 (Certified CG algorithm) time: {time.time() - step4_start:.4f}s")
        
        # Step 5: Update model parameters
        step5_start = time.time()
        self._clip_and_update_parameters(unlearned_model, param_changes)
        print(f"Step 5 (Parameter update) time: {time.time() - step5_start:.4f}s")
        
        unlearning_time = time.time() - start_time
        print(f"Total SGU unlearning time: {unlearning_time:.4f}s")
        
        return unlearning_time, unlearned_model
