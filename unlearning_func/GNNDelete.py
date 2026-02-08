import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, negative_sampling
from tqdm import tqdm, trange

from logger import get_logger
from utils import get_model

logger = get_logger('GNNDelete')

class DeletionLayer(nn.Module):
    """Deletion operator layer"""
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        """Apply deletion operator only to local nodes identified by mask"""
        if mask is None:
            mask = self.mask
        
        if mask is not None and mask.any():
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)
            return new_rep
        
        return x

class GNNDeleteModel(nn.Module):
    """GNNDelete model wrapping original model with deletion operators"""
    def __init__(self, original_model, mask_1hop, mask_2hop, hidden_dim, out_dim):
        super().__init__()
        self.original_model = original_model
        
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        self.deletion1 = DeletionLayer(hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(out_dim, mask_2hop)
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
    
    def forward(self, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        """Forward pass with deletion operations"""
        if hasattr(self.original_model, 'conv1') and hasattr(self.original_model, 'conv2'):
            x = self.original_model.x_embedding
            edge_index = self.original_model.edge_index
            
            x1 = self.original_model.conv1(x, edge_index)
            x1 = self.deletion1(x1, mask_1hop)
            x1_activated = F.relu(x1)
            
            x2 = self.original_model.conv2(x1_activated, edge_index)
            x2 = self.deletion2(x2, mask_2hop)
            
            if return_all_emb:
                return x1, x2
            return x2
        else:
            output = self.original_model()
            output = self.deletion2(output, mask_2hop)
            return output
    
    def get_original_embeddings(self, return_all_emb=False):
        """Get original (undeleted) embeddings"""
        with torch.no_grad():
            if hasattr(self.original_model, 'conv1') and hasattr(self.original_model, 'conv2'):
                x = self.original_model.x_embedding
                edge_index = self.original_model.edge_index
                
                x1 = self.original_model.conv1(x, edge_index)
                x1_activated = F.relu(x1)
                x2 = self.original_model.conv2(x1_activated, edge_index)
                
                if return_all_emb:
                    return x1, x2
                return x2
            else:
                return self.original_model()
    
    def predict_link(self, node_embeddings, edges):
        """Predict links using node embeddings"""
        src_embeddings = node_embeddings[edges[:, 0]]
        dst_embeddings = node_embeddings[edges[:, 1]]
        return torch.sum(src_embeddings * dst_embeddings, dim=1)

def get_loss_function(loss_type):
    """Get loss function"""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'kld':
        def bounded_kld(logits, truth):
            return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), 
                                          truth.softmax(-1), 
                                          None, None, 'batchmean'))
        return bounded_kld
    elif loss_type == 'cosine':
        def cosine_distance(logits, truth):
            if len(logits.shape) == 1:
                return 1 - F.cosine_similarity(logits.view(1, -1), truth.view(1, -1))
            else:
                return 1 - F.cosine_similarity(logits, truth)
        return cosine_distance
    else:
        raise NotImplementedError(f"Loss function {loss_type} not implemented")

class GNNDelete:
    """GNNDelete method implementation"""
    
    def __init__(self, data, args, device):
        self.data = data
        self.args = args
        self.device = device
    
    def _determine_local_subgraph(self, unlearned_data):
        """Determine local subgraph affected by deletion"""
        if self.args.unlearning_task == 'edge':
            deleted_edges = unlearned_data['unlearned_edges'].to(self.device)
            deleted_nodes = torch.unique(deleted_edges.flatten())
            
            train_edges = self.data['train']['edges'].to(self.device)
            
            subset_1hop, _, _, _ = k_hop_subgraph(
                deleted_nodes, 1, train_edges.t(), 
                num_nodes=unlearned_data['nodes_num']
            )
            subset_2hop, _, _, _ = k_hop_subgraph(
                deleted_nodes, 2, train_edges.t(), 
                num_nodes=unlearned_data['nodes_num']
            )
            
            node_mask_1hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
            node_mask_2hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
            
            if len(subset_1hop) > 0:
                node_mask_1hop[subset_1hop] = True
            if len(subset_2hop) > 0:
                node_mask_2hop[subset_2hop] = True
            
        elif self.args.unlearning_task == 'node':
            deleted_nodes = unlearned_data['unlearned_nodes'].to(self.device)
            train_edges = self.data['train']['edges'].to(self.device)
            
            subset_1hop, _, _, _ = k_hop_subgraph(
                deleted_nodes, 1, train_edges.t(),
                num_nodes=unlearned_data['nodes_num']
            )
            subset_2hop, _, _, _ = k_hop_subgraph(
                deleted_nodes, 2, train_edges.t(),
                num_nodes=unlearned_data['nodes_num']
            )
            
            node_mask_1hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
            node_mask_2hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
            
            if len(subset_1hop) > 0:
                node_mask_1hop[subset_1hop] = True
            if len(subset_2hop) > 0:
                node_mask_2hop[subset_2hop] = True
                
        elif self.args.unlearning_task == 'node_feature':
            if 'unlearned_nodes' in unlearned_data:
                deleted_nodes = unlearned_data['unlearned_nodes'].to(self.device)
                node_mask_1hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
                node_mask_2hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)    
                if len(deleted_nodes) > 0:
                    node_mask_1hop[deleted_nodes] = True
                    node_mask_2hop[deleted_nodes] = True
            else:
                node_mask_1hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
                node_mask_2hop = torch.zeros(unlearned_data['nodes_num'], dtype=torch.bool, device=self.device)
        
        return node_mask_1hop, node_mask_2hop
    
    def _compute_deletion_losses(self, gnn_delete_model, unlearned_data, 
                               node_mask_1hop, node_mask_2hop, original_embeddings):
        """计算删除边一致性损失和邻域影响损失"""
        
        # 获取删除后的嵌入
        current_embeddings = gnn_delete_model(node_mask_1hop, node_mask_2hop)
        
        # 1. 删除边一致性损失 (Deleted Edge Consistency Loss)
        deleted_edges = unlearned_data['unlearned_edges'].to(self.device)
        
        # 生成负样本边用于随机性比较
        neg_size = len(deleted_edges)
        train_edges = self.data['train']['edges'].to(self.device)
        neg_edges = negative_sampling(
            edge_index=train_edges.t(),
            num_nodes=unlearned_data['nodes_num'],
            num_neg_samples=neg_size
        ).t()
        
        deleted_edge_scores = gnn_delete_model.predict_link(current_embeddings, deleted_edges)
        random_edge_scores = gnn_delete_model.predict_link(current_embeddings, neg_edges)
        
        loss_fct = get_loss_function(self.args.gnndelete_loss_type)
        loss_dec = loss_fct(deleted_edge_scores, random_edge_scores)
        
        affected_nodes = node_mask_2hop.nonzero().squeeze(-1)
        
        if len(affected_nodes) > 1:
            affected_nodes_cpu = affected_nodes.cpu()

            num_samples = len(affected_nodes_cpu)
            idx1 = torch.randint(0, len(affected_nodes_cpu), (num_samples,))
            idx2 = torch.randint(0, len(affected_nodes_cpu), (num_samples,))
            valid_mask = idx1 != idx2
            if valid_mask.any():
                idx1 = idx1[valid_mask]
                idx2 = idx2[valid_mask]
                node_pairs_cpu = torch.stack([affected_nodes_cpu[idx1], affected_nodes_cpu[idx2]], dim=1)
            else:
                node_pairs_cpu = torch.empty((0, 2), dtype=torch.long)
            
            if len(node_pairs_cpu) > 0:
                node_pairs = node_pairs_cpu.to(self.device)
                original_similarities = torch.sum(
                    original_embeddings[node_pairs[:, 0]] * original_embeddings[node_pairs[:, 1]], 
                    dim=1
                )
                current_similarities = torch.sum(
                    current_embeddings[node_pairs[:, 0]] * current_embeddings[node_pairs[:, 1]], 
                    dim=1
                )
                
                loss_ni = loss_fct(current_similarities, original_similarities)
            else:
                loss_ni = torch.tensor(0.0, device=self.device)
        else:
            loss_ni = torch.tensor(0.0, device=self.device)
        
        return loss_dec, loss_ni
    
    def unlearn(self, original_model, unlearned_data):
        """Execute GNNDelete unlearning process"""
        start_time = time.time()
        
        logger.info("Starting GNNDelete unlearning process...")
        
        node_mask_1hop, node_mask_2hop = self._determine_local_subgraph(unlearned_data)
        
        logger.info(f"Affected 1-hop nodes: {node_mask_1hop.sum().item()}")
        logger.info(f"Affected 2-hop nodes: {node_mask_2hop.sum().item()}")
        
        if hasattr(original_model, 'conv1'):
            hidden_dim = original_model.conv1.out_channels if hasattr(original_model.conv1, 'out_channels') else self.args.out_dim
        else:
            hidden_dim = self.args.out_dim
        
        gnn_delete_model = GNNDeleteModel(
            original_model, node_mask_1hop, node_mask_2hop, 
            hidden_dim, self.args.out_dim
        ).to(self.device)
        
        with torch.no_grad():
            original_embeddings = gnn_delete_model.get_original_embeddings()
        
        optimizer = torch.optim.Adam([
            p for p in gnn_delete_model.parameters() if p.requires_grad
        ], lr=self.args.gnndelete_lr)
        
        logger.info("Starting deletion operator training...")
        
        for epoch in trange(self.args.gnndelete_epochs, desc='GNNDelete Training'):
            gnn_delete_model.train()
            
            loss_dec, loss_ni = self._compute_deletion_losses(
                gnn_delete_model, unlearned_data, 
                node_mask_1hop, node_mask_2hop, original_embeddings
            )
            
            alpha = self.args.gnndelete_alpha
            total_loss = alpha * loss_dec + (1 - alpha) * loss_ni
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Total Loss={total_loss.item():.4f}, "
                          f"Deletion Consistency Loss={loss_dec.item():.4f}, "
                          f"Neighborhood Influence Loss={loss_ni.item():.4f}")
        
        class UnlearnedModel(nn.Module):
            def __init__(self, gnn_delete_model, node_mask_1hop, node_mask_2hop):
                super().__init__()
                self.gnn_delete_model = gnn_delete_model
                self.node_mask_1hop = node_mask_1hop
                self.node_mask_2hop = node_mask_2hop
                
            def forward(self):
                return self.gnn_delete_model(self.node_mask_1hop, self.node_mask_2hop)
        
        unlearned_model = UnlearnedModel(gnn_delete_model, node_mask_1hop, node_mask_2hop)
        
        end_time = time.time()
        unlearning_time = end_time - start_time
        
        logger.info(f"GNNDelete unlearning completed in: {unlearning_time:.2f}s")
        
        return unlearning_time, unlearned_model
