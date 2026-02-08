import os
import time
import pandas as pd
import torch
import numpy as np
from tabulate import tabulate

from parameters import parameter_parser
from data_process import DataProcessor
from models import get_model
from train import train_model
from unlearning_func import Retrain, GraphEraser, GNNDelete, GIF, IDEA, CSGU
from evaluate import UnlearningEvaluator, evaluate_unlearning_method


def set_seed(seed):
    """Set random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_results(args, results):
    """Log experimental results to CSV file"""
    results_path = args.results_path
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    results_data = vars(args).copy()
    results_data.update(results)
    
    new_results_df = pd.DataFrame([results_data])
    
    if os.path.exists(results_path):
        try:
            existing_results_df = pd.read_csv(results_path)
            results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing results file: {e}. Creating new file.")
            results_df = new_results_df
    else:
        results_df = new_results_df
    
    try:
        results_df.to_csv(results_path, index=False)
        print(f'Results saved to {results_path}')
    except Exception as e:
        print(f"Failed to save results: {e}")


def train_original_model(args, data_processor, device):
    """Train original model"""
    print("=== Training Original Model ===")
    
    data = data_processor.data
    train_mask, val_mask, test_mask = data_processor.create_train_test_split()
    
    num_features = data.x.shape[1]
    num_classes = data_processor.num_classes[args.dataset]
    
    model = get_model(args.model, num_features, num_classes, args)
    
    results = train_model(model, data, train_mask, val_mask, test_mask, args, device)
    
    print(f"Original model performance - Train: {results['train_acc']:.4f}, "
          f"Val: {results['val_acc']:.4f}, Test: {results['test_acc']:.4f}")
    
    return results['trainer'].model, results


def run_unlearning_experiment(args, data_processor, original_model, device):
    """Run unlearning experiment"""
    print(f"\n=== Running {args.unlearning_method} Unlearning ===")
    
    if args.unlearning_task == 'node':
        unlearn_nodes = data_processor.get_unlearn_nodes()
        unlearn_edges = None
    elif args.unlearning_task == 'edge':
        unlearn_nodes = None
        unlearn_edges = data_processor.get_unlearn_edges()
    else:
        raise ValueError(f"Unknown unlearning task: {args.unlearning_task}")
    
    if args.unlearning_method == 'Retrain':
        unlearning_method = Retrain(args, device)
    elif args.unlearning_method == 'GraphEraser':
        unlearning_method = GraphEraser(args, device)
    elif args.unlearning_method == 'GNNDelete':
        unlearning_method = GNNDelete(args, device)
    elif args.unlearning_method == 'GIF':
        unlearning_method = GIF(args, device)
    elif args.unlearning_method == 'IDEA':
        unlearning_method = IDEA(args, device)
    elif args.unlearning_method == 'CSGU':
        unlearning_method = CSGU(args, device)
    else:
        raise ValueError(f"Unknown unlearning method: {args.unlearning_method}")
    
    metrics, unlearned_model = evaluate_unlearning_method(
        unlearning_method, data_processor, original_model,
        unlearn_nodes, unlearn_edges, device
    )
    
    return metrics, unlearned_model


def print_results(args, original_results, unlearning_results):
    """Print experimental results"""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Unlearning Method: {args.unlearning_method}")
    print(f"Unlearning Task: {args.unlearning_task}")
    print(f"Unlearning Ratio: {args.unlearning_ratio}")
    print(f"Seed: {args.seed}")
    
    print("\n--- Original Model Performance ---")
    print(f"Train Acc: {original_results['train_acc']:.4f}")
    print(f"Val Acc: {original_results['val_acc']:.4f}")
    print(f"Test Acc: {original_results['test_acc']:.4f}")
    print(f"Training Time: {original_results['training_time']:.2f}s")
    
    print(f"\n--- {args.unlearning_method} Results ---")
    print(f"Unlearning Time: {unlearning_results['unlearning_time']:.2f}s")
    
    print("\nPerformance Metrics:")
    for metric in ['test_acc', 'test_f1_macro', 'test_f1_micro']:
        if f'{metric}_unlearned' in unlearning_results:
            original_val = unlearning_results.get(f'{metric}_original', 0)
            unlearned_val = unlearning_results.get(f'{metric}_unlearned', 0)
            retention = unlearning_results.get(f'{metric}_retention', 0)
            print(f"  {metric}: {original_val:.4f} → {unlearned_val:.4f} (retention: {retention:.4f})")
    
    print("\nUnlearning Effectiveness:")
    effectiveness_metrics = ['kl_divergence', 'normalized_entropy', 'confidence_drop', 'prediction_change_rate']
    for metric in effectiveness_metrics:
        if metric in unlearning_results:
            print(f"  {metric}: {unlearning_results[metric]:.4f}")
    
    print("\nPrivacy Protection:")
    if 'mia_auc' in unlearning_results:
        print(f"  MIA AUC: {unlearning_results['mia_auc']:.4f}")
        print(f"  MIA Acc: {unlearning_results['mia_acc']:.4f}")
    
    print("="*80)


def main():
    """Main function"""
    args = parameter_parser()
    
    args_df = pd.DataFrame(vars(args).items(), columns=['Argument', 'Value'])
    print('Experimental Configuration:')
    print(tabulate(args_df, headers='keys', tablefmt='psql'))
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    set_seed(args.seed)
    
    print(f"\n=== Loading {args.dataset} Dataset ===")
    data_processor = DataProcessor(args)
    
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    processed_data_path = os.path.join(data_dir, f"{args.dataset}_processed_{args.seed}.pkl")
    if not data_processor.load_processed_data(processed_data_path):
        data_processor.load_dataset()
        data_processor.save_processed_data(processed_data_path)
    
    original_model, original_results = train_original_model(args, data_processor, device)
    
    unlearning_results, unlearned_model = run_unlearning_experiment(
        args, data_processor, original_model, device
    )
    
    print_results(args, original_results, unlearning_results)
    
    all_results = {**original_results, **unlearning_results}
    log_results(args, all_results)
    
    print(f"\nExperiment completed successfully!")


if __name__ == "__main__":
    main()