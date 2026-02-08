import time

import torch

from logger import get_logger
from train import train
from utils import get_model

logger = get_logger('Retrain')

class Retrain:
    """Model unlearning using retraining method"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def unlearn(self, unlearned_data):
        """Execute retraining unlearning"""
        start_time = time.time()

        remaining_link_data = unlearned_data['remaining_link_data']
        nodes_num = unlearned_data['nodes_num']
        
        train_edges = remaining_link_data['train']['edges']
        val_edges = remaining_link_data['val']['edges']
        train_labels = remaining_link_data['train']['label']
        val_labels = remaining_link_data['val']['label']
        
        edge_index = torch.cat([train_edges, val_edges], dim=0)
        edge_sign = torch.cat([train_labels, val_labels], dim=0)
        edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)

        original_model = get_model(self.args.model, nodes_num, edge_index_s, self.args, self.device)
        
        optimizer = torch.optim.Adam(original_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        retrained_model = train(original_model, optimizer, remaining_link_data, self.args, self.device)
        
        end_time = time.time()
        unlearning_time = end_time - start_time
        
        return unlearning_time, retrained_model
