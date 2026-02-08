import argparse
from datetime import datetime
import os


def str2bool(v):
    """Convert a string to a boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parameter_parser():
    parser = argparse.ArgumentParser()
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'PubMed', 'CS'])
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN'])
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', type=str, default='auto')
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    default_results_path = os.path.join(results_dir, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
    
    parser.add_argument('--results_path', type=str, 
                        default=default_results_path,
                        help='Path to save the results in csv format.')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    
    # Data split parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    
    # Unlearning parameters
    parser.add_argument('--unlearning_method', type=str, default='Retrain', 
                        choices=['Retrain', 'GraphEraser', 'GNNDelete', 'GIF', 'IDEA', 'CSGU'])
    parser.add_argument('--unlearning_task', type=str, default='node', choices=['node', 'edge'])
    parser.add_argument('--unlearning_ratio', type=float, default=0.05)
    
    # GIF specific parameters
    parser.add_argument('--gif_iteration', type=int, default=100, help='Number of iterations for GIF')
    parser.add_argument('--gif_damp', type=float, default=0.01, help='Damping factor for GIF')
    parser.add_argument('--gif_scale', type=float, default=100000, help='Scaling factor for GIF')
    
    # GNNDelete specific parameters
    parser.add_argument('--gnndelete_epochs', type=int, default=100, help='Number of epochs for GNNDelete unlearning')
    parser.add_argument('--gnndelete_lr', type=float, default=0.01, help='Learning rate for GNNDelete deletion operators')
    parser.add_argument('--gnndelete_alpha', type=float, default=0.5, help='Trade-off parameter between randomness and locality loss')
    parser.add_argument('--gnndelete_loss_type', type=str, default='mse', choices=['mse', 'kld', 'cosine'], help='Loss function type for GNNDelete')
    
    # IDEA specific parameters
    parser.add_argument('--idea_iteration', type=int, default=100, help='Number of iterations for IDEA approximation')
    parser.add_argument('--idea_damp', type=float, default=0.01, help='Damping factor for IDEA')
    parser.add_argument('--idea_scale', type=float, default=100000, help='Scaling factor for IDEA')
    parser.add_argument('--idea_gaussian_std', type=float, default=0.01, help='Standard deviation for Gaussian noise in IDEA')
    parser.add_argument('--idea_gaussian_mean', type=float, default=0.0, help='Mean for Gaussian noise in IDEA')
    parser.add_argument('--idea_l', type=float, default=1.0, help='Lipschitz constant for IDEA certification')
    parser.add_argument('--idea_lambda', type=float, default=0.01, help='Strong convexity parameter for IDEA')
    parser.add_argument('--idea_c', type=float, default=1.0, help='Loss bound for IDEA certification')
    
    # CSGU specific parameters (modified for homophilic graphs)
    parser.add_argument('--csgu_alpha', type=float, default=0.5, help='Balance parameter for degree-based influence weight')
    parser.add_argument('--csgu_expansion_depth', type=int, default=1, help='Neighborhood expansion depth')
    parser.add_argument('--csgu_cg_iterations', type=int, default=20, help='Conjugate gradient iterations')
    parser.add_argument('--csgu_damping', type=float, default=0.1, help='CG damping parameter')
    parser.add_argument('--csgu_hessian_scale', type=float, default=1.0, help='Hessian scaling parameter')
    parser.add_argument('--csgu_update_scale', type=float, default=0.1, help='Parameter update scaling factor')
    parser.add_argument('--csgu_epsilon', type=float, default=1.0, help='Privacy budget')
    parser.add_argument('--csgu_delta', type=float, default=1e-5, help='Failure probability')
    parser.add_argument('--csgu_clip_threshold', type=float, default=1.0, help='Gradient clipping threshold')
    
    # GraphEraser specific parameters
    parser.add_argument('--eraser_num_shards', type=int, default=10, help='Number of shards for GraphEraser')
    parser.add_argument('--eraser_aggregation', type=str, default='mean', choices=['mean', 'max', 'min'], 
                        help='Aggregation method for GraphEraser')
    
    return parser.parse_args()