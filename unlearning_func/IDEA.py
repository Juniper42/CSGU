import copy
import time
import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from utils import find_k_hops
from logger import get_logger

logger = get_logger('IDEA')

class IDEA:
    def __init__(self, data, args, device):
        """IDEA: A Flexible Framework of Certified Unlearning for Graph Neural Networks"""
        self.data = data
        self.args = args
        self.device = device
        
        self.iterations = getattr(args, 'idea_iteration', 100)
        self.damp = getattr(args, 'idea_damp', 0.01)
        self.scale = getattr(args, 'idea_scale', 100000)
        self.gaussian_std = getattr(args, 'idea_gaussian_std', 0.01)
        self.gaussian_mean = getattr(args, 'idea_gaussian_mean', 0.0)
        
        self.l = getattr(args, 'idea_l', 1.0)
        self.lambda_param = getattr(args, 'idea_lambda', 0.01)
        self.c = getattr(args, 'idea_c', 1.0)
        
        self.deleted_nodes = np.array([])
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])
        self.samples_to_be_unlearned = 0.0
        
        self.certification_alpha1 = 0.0
        self.certification_alpha2 = 0.0

    def _determine_influence_zone(self, unlearned_data):
        """Determine influence zone based on IDEA's find_k_hops method"""
        args = self.args
        
        if args.unlearning_task == 'edge':
            deleted_edges = unlearned_data['unlearned_edges']
            deleted_nodes = torch.unique(deleted_edges.flatten())
            k = args.layer_num
            influence_nodes = find_k_hops(deleted_nodes, k, self.data['train']['edges'])
            
            self.influence_nodes = influence_nodes.cpu().numpy()
            self.samples_to_be_unlearned = 0.0
            
        elif args.unlearning_task == 'node':
            deleted_nodes = unlearned_data['unlearned_nodes']
            k = args.layer_num
            influence_nodes = find_k_hops(deleted_nodes, k + 1, self.data['train']['edges'])
            
            self.deleted_nodes = deleted_nodes.cpu().numpy()
            self.influence_nodes = influence_nodes.cpu().numpy()
            self.samples_to_be_unlearned = float(len(deleted_nodes))
            
        elif args.unlearning_task == 'node_feature':
            if 'unlearned_nodes' in unlearned_data:
                deleted_nodes = unlearned_data['unlearned_nodes']
                self.deleted_nodes = deleted_nodes.cpu().numpy()
                self.influence_nodes = deleted_nodes.cpu().numpy()
                self.samples_to_be_unlearned = float(len(deleted_nodes))
            else:
                self.deleted_nodes = np.array([])
                self.influence_nodes = np.array([])
                self.samples_to_be_unlearned = 0.0
            
        return self.influence_nodes

    def _calculate_gradient_differences(self, model, unlearned_data, influence_nodes):
        """Calculate gradient differences - core of IDEA method based on loss difference between original and unlearned graphs"""
        model.eval()
        
        full_train_edges = self.data['train']['edges'].to(self.device)
        full_train_labels = self.data['train']['label'].to(self.device)
        
        remaining_edges = unlearned_data['remaining_link_data']['train']['edges'].to(self.device)
        remaining_labels = unlearned_data['remaining_link_data']['train']['label'].to(self.device)
        
        node_embeddings = model()
        src_embeddings = node_embeddings[full_train_edges[:, 0]]
        dst_embeddings = node_embeddings[full_train_edges[:, 1]]
        out_all = torch.sum(src_embeddings * dst_embeddings, dim=1)
        loss_all = F.binary_cross_entropy_with_logits(out_all, full_train_labels.float(), reduction='sum')
        
        if self.args.unlearning_task == 'edge':
            affected_mask = torch.isin(full_train_edges, torch.from_numpy(influence_nodes).to(self.device)).any(dim=1)
        elif self.args.unlearning_task == 'node':
            all_affected_nodes = torch.cat([
                torch.from_numpy(self.deleted_nodes).to(self.device),
                torch.from_numpy(influence_nodes).to(self.device)
            ])
            affected_mask = torch.isin(full_train_edges, all_affected_nodes).any(dim=1)
        elif self.args.unlearning_task == 'node_feature':
            # 对于节点特征遗忘，计算包含影响节点的边
            if len(influence_nodes) > 0:
                affected_mask = torch.isin(full_train_edges, torch.from_numpy(influence_nodes).to(self.device)).any(dim=1)
            else:
                affected_mask = torch.zeros(len(full_train_edges), dtype=torch.bool, device=self.device)
        
        affected_out = out_all[affected_mask]
        affected_labels = full_train_labels[affected_mask]
        loss_affected_original = F.binary_cross_entropy_with_logits(affected_out, affected_labels.float(), reduction='sum')
        
        if len(remaining_edges) > 0:
            remaining_mask = torch.isin(remaining_edges, torch.from_numpy(influence_nodes).to(self.device)).any(dim=1)
            if remaining_mask.any():
                remaining_affected_edges = remaining_edges[remaining_mask]
                remaining_affected_labels = remaining_labels[remaining_mask]
                
                src_emb_remaining = node_embeddings[remaining_affected_edges[:, 0]]
                dst_emb_remaining = node_embeddings[remaining_affected_edges[:, 1]]
                out_remaining = torch.sum(src_emb_remaining * dst_emb_remaining, dim=1)
                loss_affected_remaining = F.binary_cross_entropy_with_logits(
                    out_remaining, remaining_affected_labels.float(), reduction='sum')
            else:
                loss_affected_remaining = torch.sum(node_embeddings * 0.0)
        else:
            loss_affected_remaining = torch.sum(node_embeddings * 0.0)
        
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        grad_all = grad(loss_all, model_params, create_graph=True, retain_graph=True, allow_unused=True)
        grad_affected_original = grad(loss_affected_original, model_params, create_graph=True, retain_graph=True, allow_unused=True)
        grad_affected_remaining = grad(loss_affected_remaining, model_params, create_graph=True, retain_graph=True, allow_unused=True)
        
        grad_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_all, model_params)]
        grad_affected_original = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_affected_original, model_params)]
        grad_affected_remaining = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_affected_remaining, model_params)]
        
        return grad_all, grad_affected_original, grad_affected_remaining

    def _hvp(self, grad_all, model_params, h_estimate):
        """Calculate Hessian-vector product for iterative Hessian inverse approximation"""
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem.detach())
        
        return_grads = grad(element_product, model_params, create_graph=True, allow_unused=True)
        # 处理None梯度的情况
        return_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(return_grads, model_params)]
        return return_grads

    def _approximate_parameter_change(self, grad_tuple):
        """
        使用IDEA的迭代方法近似参数变化
        grad_tuple = (grad_all, grad_affected_original, grad_affected_remaining)
        """
        start_time = time.time()
        
        v = tuple(g1 - g2 for g1, g2 in zip(grad_tuple[1], grad_tuple[2]))
        h_estimate = tuple(g1 - g2 for g1, g2 in zip(grad_tuple[1], grad_tuple[2]))
        
        model_params = [p for p in self.original_model.parameters() if p.requires_grad]
        
        for iteration in range(self.iterations):
            hv = self._hvp(grad_tuple[0], model_params, h_estimate)
            
            with torch.no_grad():
                h_estimate = [v1 + (1 - self.damp) * h_est - hv1 / self.scale
                             for v1, h_est, hv1 in zip(v, h_estimate, hv)]
            
            h_norm = sum(torch.norm(hi).item() for hi in h_estimate)
            if torch.isnan(torch.tensor(h_norm)) or h_norm > 1e6:
                logger.warning(f"Numerical instability detected at iteration {iteration}, breaking early")
                break
        
        params_change = [h_est / self.scale for h_est in h_estimate]
        
        end_time = time.time()
        approximation_time = end_time - start_time
        
        return params_change, approximation_time

    def _add_gaussian_noise(self, params_change):
        """Add Gaussian noise for certified unlearning"""
        gaussian_noise = [
            (torch.randn(param.size(), device=param.device) * self.gaussian_std + self.gaussian_mean)
            for param in params_change
        ]
        
        noisy_params_change = [p + n for p, n in zip(params_change, gaussian_noise)]
        return noisy_params_change

    def _apply_parameter_change(self, model, params_change):
        """Apply parameter changes to model"""
        unlearned_model = copy.deepcopy(model)
        
        param_idx = 0
        with torch.no_grad():
            for param in unlearned_model.parameters():
                if param.requires_grad:
                    change = params_change[param_idx]
                    
                    if torch.isnan(change).any():
                        logger.warning(f"NaN detected in parameter change {param_idx}, setting to zero")
                        change = torch.zeros_like(change)
                    
                    param.data += change
                    param_idx += 1
        
        return unlearned_model

    def _compute_certification_bounds(self, params_change):
        """Calculate IDEA certification bounds"""
        m = self.samples_to_be_unlearned
        t = len(self.influence_nodes)
        
        if m > 0:  # Node unlearning
            numerator = m * self.l + np.sqrt(m**2 * self.l**2 + 4 * self.lambda_param * len(self.data['train']['edges']) * t * self.c)
            self.certification_alpha1 = numerator / (self.lambda_param * len(self.data['train']['edges']))
        else:  # Edge unlearning
            self.certification_alpha1 = 0.0
        
        params_change_flatten = [param.flatten() for param in params_change]
        self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2).item()
        
        total_bound = self.certification_alpha1 + self.certification_alpha2
        
        logger.info(f"Certification alpha1 (theoretical bound): {self.certification_alpha1:.6f}")
        logger.info(f"Certification alpha2 (L2 norm of change): {self.certification_alpha2:.6f}")
        logger.info(f"Total certification bound: {total_bound:.6f}")
        
        return total_bound

    def unlearn(self, original_model, unlearned_data):
        """Execute IDEA unlearning method"""
        start_time = time.time()
        
        self.original_model = original_model
        
        logger.info(f"Starting IDEA unlearning for task: {self.args.unlearning_task}")
        
        influence_nodes = self._determine_influence_zone(unlearned_data)
        logger.info(f"Influence zone determined: {len(influence_nodes)} nodes affected")
        
        grad_tuple = self._calculate_gradient_differences(original_model, unlearned_data, influence_nodes)
        logger.info("Gradient differences calculated")
        
        params_change, approx_time = self._approximate_parameter_change(grad_tuple)
        logger.info(f"Parameter change approximated in {approx_time:.4f}s using {self.iterations} iterations")
        
        noisy_params_change = self._add_gaussian_noise(params_change)
        logger.info(f"Gaussian noise added (std={self.gaussian_std})")
        
        unlearned_model = self._apply_parameter_change(original_model, noisy_params_change)
        logger.info("Parameter changes applied to model")
        
        certification_bound = self._compute_certification_bounds(params_change)
        
        end_time = time.time()
        unlearning_time = end_time - start_time
        
        logger.info(f"IDEA unlearning completed in {unlearning_time:.4f}s")
        
        return unlearning_time, unlearned_model