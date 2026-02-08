import copy
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans

from evaluate import evaluate
from logger import get_logger
from train import train
from utils import get_model

logger = get_logger('GraphEraser')

class GraphEraser:
    def __init__(self, link_data, args, device):
        self.link_data = link_data
        self.args = args
        self.device = device
        self.num_shards = args.num_shards
        
    def unlearn(self, unlearned_data):
        """Execute GraphEraser unlearning operation"""
        logger.info(f"Starting GraphEraser unlearning with {self.num_shards} shards")
        start_time = time.time()
        
        logger.info("Step 1: Performing graph partitioning...")
        shard_data_list = self._graph_partition(unlearned_data)
        logger.info(f"Graph partitioning complete, generated {len(shard_data_list)} valid shards")
        
        logger.info("Step 2: Training shard models...")
        shard_models = self._train_shard_models(shard_data_list)
        logger.info(f"Shard model training complete, successfully trained {len(shard_models)} models")
        
        logger.info("Step 3: Creating aggregated model...")
        unlearned_model = AggregatedModel(shard_models, self.device)
        
        logger.info("Step 4: Updating aggregation weights...")
        unlearned_model.update_aggregation_weights(self.link_data['val'])
        
        end_time = time.time()
        unlearning_time = end_time - start_time
        logger.info(f"GraphEraser unlearning complete, total time: {unlearning_time:.2f}s")
        
        return unlearning_time, unlearned_model
    
    def _graph_partition(self, unlearned_data):
        """Graph partitioning: divide training data into multiple shards using GraphEraser-BLPA method"""
        train_edges = self.link_data['train']['edges']
        train_labels = self.link_data['train']['label']
        
        unique_nodes = torch.unique(train_edges.flatten())
        num_nodes = len(unique_nodes)
        node_mapping = {node.item(): i for i, node in enumerate(unique_nodes)}
        
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=self.device)
        
        train_edges_gpu = train_edges.to(self.device)
        src_indices = torch.tensor([node_mapping[edge[0].item()] for edge in train_edges], device=self.device)
        dst_indices = torch.tensor([node_mapping[edge[1].item()] for edge in train_edges], device=self.device)
        
        adj_matrix[src_indices, dst_indices] = True
        adj_matrix[dst_indices, src_indices] = True
        
        node_communities = self._constrained_label_propagation(adj_matrix, num_nodes)
        
        shard_data_list = []
        
        train_edges_gpu = train_edges.to(self.device)
        train_labels_gpu = train_labels.to(self.device)
        
        src_indices = torch.tensor([node_mapping[edge[0].item()] for edge in train_edges], device=self.device)
        dst_indices = torch.tensor([node_mapping[edge[1].item()] for edge in train_edges], device=self.device)
        node_communities_gpu = torch.tensor(node_communities, device=self.device)
        
        for shard_id in range(self.num_shards):
            src_in_shard = (node_communities_gpu[src_indices] == shard_id)
            dst_in_shard = (node_communities_gpu[dst_indices] == shard_id)
            edge_mask = src_in_shard | dst_in_shard
            
            if torch.sum(edge_mask) > 0:
                shard_edges = train_edges_gpu[edge_mask]
                shard_labels = train_labels_gpu[edge_mask]
                
                if 'unlearn_indices' in unlearned_data and unlearned_data['unlearn_indices'] is not None:
                    unlearn_edges = unlearned_data['unlearned_edges'].to(self.device)
                    
                    keep_mask = torch.ones(len(shard_edges), dtype=torch.bool, device=self.device)
                    
                    for unlearn_edge in unlearn_edges:
                        edge_matches = torch.all(shard_edges == unlearn_edge.unsqueeze(0), dim=1)
                        keep_mask &= ~edge_matches
                    
                    shard_edges = shard_edges[keep_mask]
                    shard_labels = shard_labels[keep_mask]
                
                if len(shard_edges) > 0:
                    val_data = {
                        'edges': self.link_data['val']['edges'].to(self.device),
                        'label': self.link_data['val']['label'].to(self.device)
                    }
                    test_data = {
                        'edges': self.link_data['test']['edges'].to(self.device),
                        'label': self.link_data['test']['label'].to(self.device)
                    }
                    
                    shard_data = {
                        'train': {'edges': shard_edges, 'label': shard_labels},
                        'val': val_data,
                        'test': test_data
                    }
                    shard_data_list.append(shard_data)
                    logger.info(f"Shard {shard_id} created successfully, contains {len(shard_edges)} edges")
                else:
                    logger.warning(f"Shard {shard_id} is empty after removing unlearned edges, skipping")
        
        return shard_data_list
    
    def _constrained_label_propagation(self, adj_matrix, num_nodes):
        """Constrained label propagation algorithm ensuring balanced shard sizes"""
        node_communities = torch.randint(0, self.num_shards, (num_nodes,), device=self.device)
        node_threshold = num_nodes // self.num_shards + 1
        
        max_iterations = 10
        for iteration in range(max_iterations):
            old_communities = node_communities.clone()
            
            node_order = torch.randperm(num_nodes, device=self.device)
            
            for node in node_order:
                neighbors = torch.where(adj_matrix[node])[0]
                if len(neighbors) == 0:
                    continue
                
                neighbor_communities = node_communities[neighbors]
                unique_communities, counts = torch.unique(neighbor_communities, return_counts=True)
                
                if len(unique_communities) > 0:
                    max_count_idx = torch.argmax(counts)
                    desired_community = unique_communities[max_count_idx].item()
                    
                    current_community = node_communities[node].item()
                    if desired_community != current_community:
                        target_size = torch.sum(node_communities == desired_community).item()
                        if target_size < node_threshold:
                            node_communities[node] = desired_community
            
            if torch.equal(old_communities, node_communities):
                break
        
        return node_communities.cpu().numpy()
    
    def _train_shard_models(self, shard_data_list):
        """Train independent sub-models on each data shard"""
        shard_models = []
        
        all_train_edges = self.link_data['train']['edges']
        all_val_edges = self.link_data['val']['edges']
        all_test_edges = self.link_data['test']['edges']
        
        global_edges = torch.cat([all_train_edges, all_val_edges, all_test_edges], dim=0)
        global_nodes_num = torch.unique(global_edges.flatten()).max().item() + 1
        logger.info(f"Global node count: {global_nodes_num}")
        
        for i, shard_data in enumerate(shard_data_list):
            logger.info(f"Training shard model {i+1}/{len(shard_data_list)}")
            
            if len(shard_data['train']['edges']) == 0:
                logger.warning(f"Shard {i} has no training data, skipping")
                continue
            
            all_edges = torch.cat([
                shard_data['train']['edges'].to(self.device), 
                shard_data['val']['edges'].to(self.device)
            ], dim=0)
            all_labels = torch.cat([
                shard_data['train']['label'].to(self.device), 
                shard_data['val']['label'].to(self.device)
            ], dim=0)
    
            if len(all_edges) == 0:
                logger.warning(f"Shard {i} has empty edge data after merging, skipping")
                continue
            
            nodes_num = global_nodes_num
            
            if all_edges.shape[0] == 0 or all_edges.shape[1] != 2:
                logger.warning(f"Shard {i} has invalid edge data format, skipping")
                continue
            
            if len(all_labels) != len(all_edges):
                logger.warning(f"Shard {i} label length doesn't match edge data, skipping")
                continue
                
            edge_index_s = torch.cat([all_edges, all_labels.unsqueeze(-1)], dim=-1)
            
            if edge_index_s.shape[0] == 0:
                logger.warning(f"Shard {i} has empty final edge index, skipping")
                continue
                
            max_node_id = edge_index_s[:, :2].max().item()
            if max_node_id >= nodes_num:
                logger.warning(f"Shard {i} contains invalid node ID {max_node_id} >= {nodes_num}, skipping")
                continue
            
            logger.info(f"Shard {i} edge index statistics:")
            logger.info(f"  - Edge index shape: {edge_index_s.shape}")
            logger.info(f"  - Min node ID: {edge_index_s[:, :2].min().item()}")
            logger.info(f"  - Max node ID: {max_node_id}")
            logger.info(f"  - Label distribution: {torch.unique(edge_index_s[:, 2], return_counts=True)}")
            
            pos_edges = edge_index_s[edge_index_s[:, 2] > 0]
            neg_edges = edge_index_s[edge_index_s[:, 2] < 0]
            
            if len(pos_edges) == 0:
                logger.warning(f"Shard {i} has no positive edges, adding some")
                if len(neg_edges) > 0:
                    sample_size = min(10, len(neg_edges))
                    sample_indices = torch.randperm(len(neg_edges))[:sample_size]
                    sampled_neg_edges = neg_edges[sample_indices]
                    pos_edges = sampled_neg_edges.clone()
                    pos_edges[:, 2] = 1
                    edge_index_s = torch.cat([edge_index_s, pos_edges], dim=0)
            
            if len(neg_edges) == 0:
                logger.warning(f"Shard {i} has no negative edges, adding some")
                if len(pos_edges) > 0:
                    sample_size = min(10, len(pos_edges))
                    sample_indices = torch.randperm(len(pos_edges))[:sample_size]
                    sampled_pos_edges = pos_edges[sample_indices]
                    neg_edges = sampled_pos_edges.clone()
                    neg_edges[:, 2] = -1
                    edge_index_s = torch.cat([edge_index_s, neg_edges], dim=0)
            
            min_edges_required = 20
            if len(edge_index_s) < min_edges_required:
                logger.warning(f"Shard {i} has insufficient edges ({len(edge_index_s)} < {min_edges_required}), skipping")
                continue
            
            logger.info(f"Shard {i}: edges={len(all_edges)}, nodes={nodes_num}")
            
            try:
                shard_model = get_model(self.args.model, nodes_num, edge_index_s, self.args, self.device)
                
                with torch.no_grad():
                    try:
                        _ = shard_model()
                    except Exception as e:
                        logger.error(f"Shard {i} model forward pass failed: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error creating model for shard {i}: {e}")
                logger.error(f"Edge data shape: {edge_index_s.shape}")
                logger.error(f"Edge data content: {edge_index_s[:10]}")
                continue
            shard_optimizer = torch.optim.Adam(shard_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            
            reduced_args = copy.deepcopy(self.args)
            reduced_args.epochs = min(self.args.epochs // 2, 50)
            reduced_args.patience = min(self.args.patience, 10)
            
            shard_model = train(shard_model, shard_optimizer, shard_data, reduced_args, self.device)
            shard_models.append(shard_model)
            logger.info(f"Shard model {i+1} training successful")
        
        return shard_models


class AggregatedModel:
    """Aggregate predictions from multiple sub-models using learned aggregation"""
    def __init__(self, shard_models, device):
        self.shard_models = shard_models
        self.device = device
        self.num_shards = len(shard_models)
        
        self.aggregation_weights = torch.ones(self.num_shards, device=device) / self.num_shards
        
        if self.shard_models:
            sample_embedding = self.shard_models[0]()
            self.embedding_dim = sample_embedding.shape[1] if len(sample_embedding.shape) > 1 else sample_embedding.shape[0]
            self.max_nodes = sample_embedding.shape[0]
        else:
            self.embedding_dim = 64
            self.max_nodes = 100
        
    def __call__(self):
        """Forward pass: aggregate embeddings from all sub-models using learned aggregation"""
        self.eval()
        embeddings_list = []
        
        for model in self.shard_models:
            with torch.no_grad():
                embedding = model()
                embeddings_list.append(embedding)
        
        if len(embeddings_list) == 0:
            return torch.zeros((self.max_nodes, self.embedding_dim), device=self.device)
        
        target_nodes = self.max_nodes
        normalized_embeddings = []
        
        for embedding in embeddings_list:
            embedding = embedding.to(self.device)
            
            if embedding.shape[0] < target_nodes:
                padding = torch.zeros((target_nodes - embedding.shape[0], embedding.shape[1]), 
                                    device=self.device, dtype=embedding.dtype)
                embedding = torch.cat([embedding, padding], dim=0)
            elif embedding.shape[0] > target_nodes:
                embedding = embedding[:target_nodes]
            
            normalized_embeddings.append(embedding)
        
        embeddings_stack = torch.stack(normalized_embeddings)
        
        weighted_embeddings = embeddings_stack * self.aggregation_weights.view(-1, 1, 1)
        aggregated_embeddings = torch.sum(weighted_embeddings, dim=0)
        
        return aggregated_embeddings
    
    def update_aggregation_weights(self, validation_data=None):
        """Update aggregation weights based on validation data"""
        if validation_data is None or len(self.shard_models) <= 1:
            return
        
        shard_performances = []
        
        for model in self.shard_models:
            model.eval()
            embedding = model()
            performance = 1.0 / (1.0 + torch.norm(embedding).item())
            shard_performances.append(performance)
        
        performances_tensor = torch.tensor(shard_performances, device=self.device)
        self.aggregation_weights = torch.softmax(performances_tensor, dim=0)
    
    def loss(self):
        """Calculate aggregated loss (average of all sub-model losses)"""
        losses = []
        for model in self.shard_models:
            loss = model.loss()
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return torch.stack(losses).mean()
    
    def eval(self):
        """Set all sub-models to evaluation mode"""
        for model in self.shard_models:
            model.eval()
    
    def train(self):
        """Set all sub-models to training mode"""
        for model in self.shard_models:
            model.train()
    
    def to(self, device):
        """Move all sub-models to specified device"""
        for model in self.shard_models:
            model.to(device)
        return self
    
    def parameters(self):
        """Return parameters of all sub-models"""
        params = []
        for model in self.shard_models:
            params.extend(list(model.parameters()))
        return params