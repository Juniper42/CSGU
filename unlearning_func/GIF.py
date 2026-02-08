import copy
import time

import torch
import torch.nn.functional as F
from torch.autograd import grad

from utils import find_k_hops


class GIF:
    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
        self.iterations = args.gif_iteration
        self.damp = args.gif_damp
        self.scale = args.gif_scale

    def _determine_influence_zone(self, unlearned_data):
        """Determines the influence zone based on the unlearning task."""
        args = self.args
        if args.unlearning_task == 'edge':
            k = args.layer_num
            deleted_nodes = torch.unique(unlearned_data['unlearned_edges'].flatten())
            influence_nodes = find_k_hops(deleted_nodes, k, self.data['train']['edges'])
        elif args.unlearning_task == 'node':
            k = args.layer_num
            deleted_nodes = unlearned_data['unlearned_nodes']
            influence_nodes = find_k_hops(deleted_nodes, k + 1, self.data['train']['edges'])
        elif args.unlearning_task == 'node_feature':
            if 'unlearned_nodes' in unlearned_data:
                influence_nodes = unlearned_data['unlearned_nodes']
            else:
                influence_nodes = torch.tensor([], dtype=torch.long)
        
        return influence_nodes.to(self.device)

    def _calculate_gradient_differences(self, unlearned_model, unlearned_data, influence_nodes):
        """Calculates the gradient differences required for GIF."""
        full_train_edges = self.data['train']['edges'].to(self.device)
        full_train_labels = self.data['train']['label'].to(self.device)
        remaining_edges = unlearned_data['remaining_link_data']['train']['edges'].to(self.device)
        remaining_labels = unlearned_data['remaining_link_data']['train']['label'].to(self.device)
        
        # Get node embeddings from the model
        node_embeddings = unlearned_model()
        
        # Calculate edge predictions for full training set
        src_embeddings = node_embeddings[full_train_edges[:, 0]]
        dst_embeddings = node_embeddings[full_train_edges[:, 1]]
        out_all = torch.sum(src_embeddings * dst_embeddings, dim=1)
        loss_all = F.binary_cross_entropy_with_logits(out_all, full_train_labels.float(), reduction='sum')

        original_affected_mask = torch.isin(full_train_edges, influence_nodes).any(dim=1)
        affected_edges_original = full_train_edges[original_affected_mask]
        affected_labels_original = full_train_labels[original_affected_mask]
        
        loss_original_affected = self._calculate_loss_on_edges(
            unlearned_model, affected_edges_original, affected_labels_original
        )

        remaining_affected_mask = torch.isin(remaining_edges, influence_nodes).any(dim=1)
        affected_edges_remaining = remaining_edges[remaining_affected_mask]
        affected_labels_remaining = remaining_labels[remaining_affected_mask]

        loss_remaining_affected = self._calculate_loss_on_edges(
            unlearned_model, affected_edges_remaining, affected_labels_remaining
        )

        model_params = [p for p in unlearned_model.parameters() if p.requires_grad]
        
        grad_all_tuple = grad(loss_all, model_params, retain_graph=True, create_graph=True, allow_unused=True)
        grad_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_all_tuple, model_params)]
        
        grad_original_affected = self._get_grads(loss_original_affected, model_params)
        grad_remaining_affected = self._get_grads(loss_remaining_affected, model_params)
        
        v = [g1 - g2 for g1, g2 in zip(grad_original_affected, grad_remaining_affected)]
        
        return v, grad_all, model_params

    def _estimate_and_update_model(self, unlearned_model, v, grad_all, model_params):
        """Estimates parameter changes using Hessian-vector products and updates the model."""
        h_estimate = v
        iterations = self.iterations
        damp = self.damp
        scale = self.scale

        for i in range(iterations):
            hv = self.hvps(grad_all, model_params, h_estimate)
            with torch.no_grad():
                h_estimate = [v_i + (1 - damp) * h_i - hv_i / scale for v_i, h_i, hv_i in zip(v, h_estimate, hv)]
            
            # Check for numerical instability and break if detected
            h_norm = sum(torch.norm(hi).item() for hi in h_estimate)
            if torch.isnan(torch.tensor(h_norm)) or h_norm > 1e6:
                break

        param_change = [h_e / scale for h_e in h_estimate]
        
        # Clip parameter changes to prevent extreme updates
        # max_change = 1  # Maximum allowed parameter change
        # for i, pc in enumerate(param_change):
        #     pc_norm = torch.norm(pc)
        #     if pc_norm > max_change:
        #         param_change[i] = pc * (max_change / pc_norm)

        # with torch.no_grad():
        #     for p, change in zip(model_params, param_change):
        #         p.add_(change)

        with torch.no_grad():
            # max_change = 1
            for p, h_e in zip(model_params, h_estimate):
                change = h_e / scale
                # change_norm = change.norm()
                # if change_norm > max_change:
                #     change.mul_(max_change / change_norm)
                p.add_(change)
        
        return unlearned_model

    def _calculate_loss_on_edges(self, model, edges, labels):
        if edges.size(0) > 0:
            node_embeddings = model()
            src_embeddings = node_embeddings[edges[:, 0]]
            dst_embeddings = node_embeddings[edges[:, 1]]
            out = torch.sum(src_embeddings * dst_embeddings, dim=1)
            return F.binary_cross_entropy_with_logits(out, labels.float(), reduction='sum')
        return torch.tensor(0.0, device=self.device)

    def _get_grads(self, loss, model_params):
        if loss.requires_grad:
            grads_tuple = grad(loss, model_params, retain_graph=True, create_graph=True, allow_unused=True)
            return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_tuple, model_params)]
        return [torch.zeros_like(p) for p in model_params]

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            # Ensure v_elem doesn't require grad
            element_product += torch.sum(grad_elem * v_elem.detach())
        
        hv_tuple = grad(element_product, model_params, create_graph=True, allow_unused=True)
        return [h if h is not None else torch.zeros_like(p) for h, p in zip(hv_tuple, model_params)]
    
    def unlearn(self, original_model, unlearned_data):
        unlearned_model = copy.deepcopy(original_model)

        start_time = time.time()

        # Step 1: Determine the influence zone based on the unlearning task
        influence_nodes = self._determine_influence_zone(unlearned_data)
        
        # Step 2: Calculate the difference in gradients
        v, grad_all, model_params = self._calculate_gradient_differences(
            unlearned_model, unlearned_data, influence_nodes
        )

        # Step 3: Estimate the parameter changes and update the model
        unlearned_model = self._estimate_and_update_model(
            unlearned_model, v, grad_all, model_params
        )
        
        unlearning_time = time.time() - start_time
        return unlearning_time, unlearned_model
