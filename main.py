import os
import os.path as osp

import pandas as pd
import torch
import numpy as np
from tabulate import tabulate
from torch_geometric_signed_directed.data.signed import load_signed_real_data
from torch_geometric.utils import degree

from data_process import train_test_gen
from evaluate import evaluate, mia
from logger import get_logger, setup_logging
from parameters import parameter_parser
from train import train
from unlearning_func import GIF, GNNDelete, GraphEraser, Retrain, CSGU, IDEA
from utils import get_model

setup_logging()
logger = get_logger('Main')

def generate_node_features(num_nodes, edge_index, feature_dim, device='cpu'):
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    base_features = torch.log(deg + 1).unsqueeze(1).repeat(1, feature_dim)
    noise = torch.randn(num_nodes, feature_dim, device=device) * 0.2
    node_features = base_features + noise
    
    node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
    
    return node_features

def log_results(args, unlearning_t_info, mia_info, unlearning_time):
    """Logs the results of the experiment to an CSV file."""
    results_path = args.results_path
    
    # Ensure the directory for the results file exists
    if not osp.exists(osp.dirname(results_path)):
        os.makedirs(osp.dirname(results_path))
    
    # Prepare the data for the new result entry
    results_data = vars(args).copy()

    results_data.update({
        'unlearning_time': unlearning_time,
        'auc': unlearning_t_info['auc'],
        'f1_macro': unlearning_t_info['f1_macro'],
        'f1_micro': unlearning_t_info['f1_micro'],
        'mia_auc': mia_info['mia_auc']
    })
    
    new_results_df = pd.DataFrame([results_data])
    
    # Read existing file or create a new DataFrame
    if osp.exists(results_path):
        try:
            existing_results_df = pd.read_csv(results_path)
            results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
        except Exception as e:
            logger.error(f"Error reading existing results file: {e}. A new file will be created.")
            results_df = new_results_df
    else:
        results_df = new_results_df
        
    # Save the updated DataFrame to the Excel file
    try:
        results_df.to_csv(results_path, index=False)
        logger.info(f'Results successfully saved to {results_path}')
    except Exception as e:
        logger.error(f"Failed to save results to {results_path}: {e}")

def main():
    args = parameter_parser()
    args_df = pd.DataFrame(vars(args).items(), columns=['Argument', 'Value'])
    logger.info('Arguments:\n' + tabulate(args_df, headers='keys', tablefmt='psql'))


    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f'Using device: {device}')

    data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset_path = osp.join(data_path, args.dataset, 'processed', str(args.seed))
    
    logger.info(f'Loading data from {args.dataset}...')
    try:
        train_path = [f for f in os.listdir(dataset_path) if f.startswith('train_') and f.endswith('.parquet')][0]
        val_path = [f for f in os.listdir(dataset_path) if f.startswith('val_') and f.endswith('.parquet')][0]
        test_path = [f for f in os.listdir(dataset_path) if f.startswith('test_') and f.endswith('.parquet')][0]
        
        logger.info(f'Found preprocessed data in {dataset_path}')
        train_df = pd.read_parquet(osp.join(dataset_path, train_path))
        val_df = pd.read_parquet(osp.join(dataset_path, val_path))
        test_df = pd.read_parquet(osp.join(dataset_path, test_path))

        train_data = {
            'edges': torch.from_numpy(train_df[['source', 'target']].values),
            'label': torch.from_numpy(train_df['label'].values)
        }
        val_data = {
            'edges': torch.from_numpy(val_df[['source', 'target']].values),
            'label': torch.from_numpy(val_df['label'].values)
        }
        test_data = {
            'edges': torch.from_numpy(test_df[['source', 'target']].values),
            'label': torch.from_numpy(test_df['label'].values)
        }

    except (FileNotFoundError, IndexError):
        logger.info('Preprocessed data not found, generating new data...')
        data = load_signed_real_data(dataset=args.dataset, root=data_path).to(device)
        train_data, val_data, test_data = train_test_gen(data, args.test_ratio, args.val_ratio, args.seed)
        save_dir = osp.join(data_path, args.dataset, 'processed', str(args.seed))
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f'Saving processed data to {save_dir}')
        for data_split, name, ratio in [(train_data, 'train', 1 - args.test_ratio - args.val_ratio), (val_data, 'val', args.val_ratio), (test_data, 'test', args.test_ratio)]:
            df = pd.DataFrame({
                'source': data_split['edges'][:, 0].cpu().numpy(),
                'target': data_split['edges'][:, 1].cpu().numpy(),
                'label': data_split['label'].cpu().numpy()
            })
            file_path = osp.join(save_dir, f'{name}_{ratio}.parquet')
            df.to_parquet(file_path)

    logger.info('Data loaded successfully.')

    for data_split in [train_data, val_data, test_data]:
        for key, value in data_split.items():
            if isinstance(value, torch.Tensor):
                data_split[key] = value.to(device)
    
    link_data = {'train': train_data, 'val': val_data, 'test': test_data}
    
    logger.info('Loading graph data for model...')
    data = load_signed_real_data(dataset=args.dataset, root=data_path).to(device)
    nodes_num = data.num_nodes
    edge_index = torch.cat([link_data['train']['edges'], link_data['val']['edges']], dim=0)
    edge_sign = torch.cat([link_data['train']['label'], link_data['val']['label']], dim=0)
    edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)
    
    if args.unlearning_task == 'node_feature' or not hasattr(data, 'x') or data.x is None:
        node_features = generate_node_features(
            num_nodes=nodes_num, 
            edge_index=edge_index.t(), 
            feature_dim=args.node_feature_dim, 
            device=device
        )
        data.x = node_features

    logger.info(f'Initializing model: {args.model}')
    model = get_model(args.model, nodes_num, edge_index_s, args, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f'Optimizer Adam initialized with lr={args.lr} and weight_decay={args.weight_decay}')

    unlearned_data = {}
    remaining_edges, remaining_labels = link_data['train']['edges'], link_data['train']['label']

    if args.unlearning_task == 'edge':
        unlearn_size = int(len(link_data['train']['edges']) * args.unlearning_ratio)
        unlearn_indices = torch.randperm(len(link_data['train']['edges']))[:unlearn_size]
        unlearned_edges = link_data['train']['edges'][unlearn_indices]
        unlearned_labels = link_data['train']['label'][unlearn_indices]
        
        unlearned_nodes = torch.unique(unlearned_edges.flatten())
        unlearned_nodes_num = len(unlearned_nodes)
        
        unlearned_edge_index_s = torch.cat([unlearned_edges, unlearned_labels.unsqueeze(-1)], dim=-1)
        
        unlearned_data = {
            'unlearn_indices': unlearn_indices,
            'unlearned_edges': unlearned_edges,
            'unlearned_labels': unlearned_labels,
            'unlearned_nodes_num': unlearned_nodes_num,
            'unlearned_edge_index_s': unlearned_edge_index_s,
            'nodes_num': nodes_num
        }
        
        mask = torch.ones(len(link_data['train']['edges']), dtype=torch.bool, device=device)
        mask[unlearn_indices.to(device)] = False
        remaining_edges = link_data['train']['edges'][mask]
        remaining_labels = link_data['train']['label'][mask]

    elif args.unlearning_task == 'node':
        unique_nodes = torch.unique(link_data['train']['edges'].flatten())
        unlearn_nodes_size = int(len(unique_nodes) * args.unlearning_ratio)
        unlearn_node_indices = torch.randperm(len(unique_nodes))[:unlearn_nodes_size]
        unlearned_nodes = unique_nodes[unlearn_node_indices]
        unlearned_nodes_num = len(unlearned_nodes)

        # Find edges connected to the unlearned nodes
        edge_mask = torch.isin(link_data['train']['edges'][:, 0], unlearned_nodes) | torch.isin(link_data['train']['edges'][:, 1], unlearned_nodes)
        
        unlearned_edges = link_data['train']['edges'][edge_mask]
        unlearned_labels = link_data['train']['label'][edge_mask]
        
        unlearned_edge_index_s = torch.cat([unlearned_edges, unlearned_labels.unsqueeze(-1)], dim=-1)
        
        unlearned_data = {
            'unlearn_indices': torch.where(edge_mask)[0],
            'unlearned_edges': unlearned_edges,
            'unlearned_labels': unlearned_labels,
            'nodes_num': unlearned_nodes_num,
            'unlearned_edge_index_s': unlearned_edge_index_s,
            'unlearned_nodes': unlearned_nodes
        }

        remaining_edges = link_data['train']['edges'][~edge_mask]
        remaining_labels = link_data['train']['label'][~edge_mask]

    elif args.unlearning_task == 'node_feature':
        unique_nodes = torch.unique(link_data['train']['edges'].flatten())
        unlearn_nodes_size = int(len(unique_nodes) * args.unlearning_ratio)
        unlearn_node_indices = torch.randperm(len(unique_nodes))[:unlearn_nodes_size]
        unlearned_nodes = unique_nodes[unlearn_node_indices]
        unlearned_nodes_num = len(unlearned_nodes)
        
        logger.info(f'Unlearning features for {unlearned_nodes_num} nodes (ratio: {args.unlearning_ratio})')
        
        original_features = data.x.clone()
        unlearned_features = data.x.clone()
        
        unlearned_features[unlearned_nodes] = 0.0
        
        unlearned_data = {
            'unlearn_indices': unlearn_node_indices,
            'unlearned_nodes': unlearned_nodes,
            'unlearned_nodes_num': unlearned_nodes_num,
            'original_features': original_features,
            'unlearned_features': unlearned_features,
            'nodes_num': nodes_num
        }
        
        remaining_edges = link_data['train']['edges']
        remaining_labels = link_data['train']['label']

    remaining_link_data = {
        'train': {'edges': remaining_edges, 'label': remaining_labels},
        'val': link_data['val'],
        'test': link_data['test']
    }
    unlearned_data['remaining_link_data'] = remaining_link_data

    for run in range(args.runs):
        logger.info(f"Run {run+1}/{args.runs} Starting training...")
        original_model = get_model(args.model, nodes_num, edge_index_s, args, device)
        optimizer = torch.optim.Adam(original_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model_cache = ".cache"
        train_ratio = 1 - args.test_ratio - args.val_ratio
        model_filename = f"{args.model}_i{args.in_dim}_o{args.out_dim}_layer{args.layer_num}_lr{args.lr}_wd{args.weight_decay}_{args.dataset}_seed{args.seed}_train{train_ratio}.pt"
        model_path = osp.join(model_cache, model_filename)
        
        if osp.exists(model_path):
            logger.info(f"Loading cached model from {model_path}")
            try:
                original_model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info("Model loaded successfully from cache")
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}. Training new model...")
                original_model = train(original_model, optimizer, link_data, args, device)
                torch.save(original_model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
        else:
            logger.info("No cached model found, training new model...")
            original_model = train(original_model, optimizer, link_data, args, device)
            os.makedirs(osp.dirname(model_path), exist_ok=True)
            torch.save(original_model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

        test_info = evaluate(original_model, link_data, device, eval_flag='test')
        logger.info(f"Before unlearning - Test Set: AUC: {test_info['auc']:.4f}, Macro F1: {test_info['f1_macro']:.4f}, Micro F1: {test_info['f1_micro']:.4f}")

        unlearning_func = None
        if args.unlearning_task != 'node_feature':
            if args.unlearning_method == 'Retrain':
                unlearning_func = Retrain(args, device)
            elif args.unlearning_method == 'SISA':
                unlearning_func = SISA(link_data, args, device)
            elif args.unlearning_method == 'GraphEraser':
                unlearning_func = GraphEraser(link_data, args, device)
            elif args.unlearning_method == 'GIF':
                unlearning_func = GIF(link_data, args, device)
            elif args.unlearning_method == 'GNNDelete':
                unlearning_func = GNNDelete(link_data, args, device)
            elif args.unlearning_method == 'IDEA':
                unlearning_func = IDEA(link_data, args, device)
            elif args.unlearning_method == 'CSGU':
                unlearning_func = CSGU(link_data, args, device)
            elif args.unlearning_method == 'CEU':
                unlearning_func = CEU(link_data, args, device)

        logger.info(f"Unlearning method: {args.unlearning_method}")
        
        if args.unlearning_task == 'node_feature':        
            unlearned_model = get_model(args.model, nodes_num, edge_index_s, args, device)
            optimizer_unlearn = torch.optim.Adam(unlearned_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            original_x = data.x.clone()
            data.x = unlearned_data['unlearned_features']
            unlearned_model = train(unlearned_model, optimizer_unlearn, link_data, args, device)
            data.x = original_x
        else:
            if args.unlearning_method in ['Retrain', 'SISA', 'GraphEraser']:
                unlearning_time, unlearned_model = unlearning_func.unlearn(unlearned_data)
            else:
                unlearning_time, unlearned_model = unlearning_func.unlearn(original_model, unlearned_data)
            
        unlearning_t_info = evaluate(unlearned_model, remaining_link_data, device, eval_flag='test')
        
        all_edges = torch.cat([link_data['train']['edges'], link_data['val']['edges'], link_data['test']['edges']], dim=0)
        mia_info = mia(unlearned_model, unlearned_data, all_edges, device)

        logger.info(f"Unlearning Time: {unlearning_time:.2f}s, AUC: {unlearning_t_info['auc']:.4f}, Macro F1: {unlearning_t_info['f1_macro']:.4f}, Micro F1: {unlearning_t_info['f1_micro']:.4f}, MIA AUC: {mia_info['mia_auc']:.4f}")

        log_results(args, unlearning_t_info, mia_info, unlearning_time)

if __name__ == "__main__":
    main() 