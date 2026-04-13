import os
import pickle
import random
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class DataProcessor:
    """Data processing class for handling homogeneous graph datasets"""
    
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.seed = args.seed
        
        self.set_seed(self.seed)
        
        self.num_features = {
            "Cora": 1433,
            "PubMed": 500, 
            "CS": 6805,
        }
        
        self.num_classes = {
            "Cora": 7,
            "PubMed": 3,
            "CS": 15,
        }
        
        self.data = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
    def set_seed(self, seed):
        """Set random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def load_dataset(self):
        """Load dataset"""
        print(f"Loading {self.dataset_name} dataset...")
        
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        
        if self.dataset_name in ["Cora", "PubMed", "CiteSeer"]:
            dataset = Planetoid(data_path, self.dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            
        elif self.dataset_name == "CS":
            dataset = Coauthor(data_path, name="CS", transform=T.NormalizeFeatures())
            data = dataset[0]
            
            if data.x.shape[1] > 1000:
                print("Applying PCA to reduce feature dimensionality...")
                features = data.x.cpu().numpy()
                n_components = min(500, features.shape[1])
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(features)
                data.x = torch.tensor(pca_result, dtype=torch.float32).to(data.x.device)
                self.num_features["CS"] = n_components
                print(f"Feature dimension reduced from {features.shape[1]} to {n_components}")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        data.edge_index = to_undirected(data.edge_index)
        self.data = data
        
        actual_num_classes = int(data.y.max().item()) + 1
        self.num_classes[self.dataset_name] = actual_num_classes
        
        print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges, "
              f"{data.x.shape[1]} features, {actual_num_classes} classes")
        
        return data
    
    def create_train_test_split(self):
        """Create train/validation/test split"""
        num_nodes = self.data.num_nodes
        indices = np.arange(num_nodes)
        
        labels = self.data.y.cpu().numpy()
        
        train_indices, temp_indices = train_test_split(
            indices, test_size=1-self.args.train_ratio, 
            stratify=labels, random_state=self.seed
        )
        
        temp_labels = labels[temp_indices]
        val_size = self.args.val_ratio / (self.args.val_ratio + self.args.test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=1-val_size,
            stratify=temp_labels, random_state=self.seed
        )
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True
        
        print(f"Train: {self.train_mask.sum()}, Val: {self.val_mask.sum()}, Test: {self.test_mask.sum()}")
        
        return self.train_mask, self.val_mask, self.test_mask
    
    def get_unlearn_nodes(self):
        """Get nodes to be unlearned"""
        train_nodes = torch.where(self.train_mask)[0]
        num_unlearn = int(len(train_nodes) * self.args.unlearning_ratio)
        
        unlearn_indices = torch.randperm(len(train_nodes))[:num_unlearn]
        unlearn_nodes = train_nodes[unlearn_indices]
        
        print(f"Selected {len(unlearn_nodes)} nodes for unlearning")
        return unlearn_nodes
    
    def get_unlearn_edges(self):
        """Get edges to be unlearned"""
        edge_index = self.data.edge_index
        num_edges = edge_index.shape[1]
        num_unlearn = int(num_edges * self.args.unlearning_ratio)
        
        edge_indices = torch.randperm(num_edges)[:num_unlearn]
        unlearn_edges = edge_index[:, edge_indices]
        
        print(f"Selected {unlearn_edges.shape[1]} edges for unlearning")
        return unlearn_edges
    
    def create_unlearn_data(self, unlearn_nodes=None, unlearn_edges=None):
        """Create data after unlearning"""
        data_unlearn = self.data.clone()
        
        if unlearn_nodes is not None:
            remaining_mask = self.train_mask.clone()
            remaining_mask[unlearn_nodes] = False
            
            return data_unlearn, remaining_mask
        
        elif unlearn_edges is not None:
            edge_index = self.data.edge_index
            edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
            
            for unlearn_edge in unlearn_edges.t():
                mask1 = (edge_index[0] == unlearn_edge[0]) & (edge_index[1] == unlearn_edge[1])
                mask2 = (edge_index[0] == unlearn_edge[1]) & (edge_index[1] == unlearn_edge[0])
                edge_mask = edge_mask & ~(mask1 | mask2)
            
            data_unlearn.edge_index = edge_index[:, edge_mask]
            print(f"Removed {(~edge_mask).sum()} edges from graph")
            
            return data_unlearn, self.train_mask
    
    def save_processed_data(self, save_path):
        """Save processed data"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data_dict = {
            'data': self.data,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'test_mask': self.test_mask,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Processed data saved to {save_path}")
    
    def load_processed_data(self, load_path):
        """Load processed data"""
        if not os.path.exists(load_path):
            return False
        
        with open(load_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = data_dict['data']
        self.train_mask = data_dict['train_mask']
        self.val_mask = data_dict['val_mask']
        self.test_mask = data_dict['test_mask']
        
        print(f"Processed data loaded from {load_path}")
        return True