import argparse
from datetime import datetime


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
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--model', type=str, default='SGCN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--in_dim', type=int)
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10, help='number of distinct runs')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='auto')
    # Path to save results
    parser.add_argument('--results_path', type=str, default=f'results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv', help='Path to save the results in xlsx format.')


    # unlearning
    parser.add_argument('--unlearning_method', type=str, default='Retrain', choices=['Retrain', 'SISA', 'GraphEraser', 'GIF', 'GNNDelete', 'IDEA', 'CEU', 'CSGU'])
    parser.add_argument('--unlearning_task', type=str, default='edge', choices=['edge', 'node', 'pos_edge', 'neg_edge', 'node_feature'])
    parser.add_argument('--unlearning_ratio', type=float, default=0.005)
    # Node feature specific parameters
    parser.add_argument('--node_feature_dim', type=int, help='Dimension of synthetic node features')
    
    # SISA specific parameters
    parser.add_argument('--num_shards', type=int, default=10, help='Number of shards for SISA method')

    # GIF specific parameters
    parser.add_argument('--gif_iteration', type=int, default=10000, help='Number of iterations for GIF')
    parser.add_argument('--gif_damp', type=float, default=0.01, help='Damping factor for GIF')
    parser.add_argument('--gif_scale', type=float, help='Scaling factor for GIF')

    # GNNDelete specific parameters
    parser.add_argument('--gnndelete_epochs', type=int, default=100, help='Number of epochs for GNNDelete unlearning')
    parser.add_argument('--gnndelete_lr', type=float, help='Learning rate for GNNDelete deletion operators')
    parser.add_argument('--gnndelete_alpha', type=float, default=0.5, help='Trade-off parameter between randomness and locality loss')
    parser.add_argument('--gnndelete_loss_type', type=str, default='mse', choices=['mse', 'kld', 'cosine'], help='Loss function type for GNNDelete')

    # CSGU specific parameters
    parser.add_argument('--csgu_alpha', type=float, default=0.5, help='Balance theory vs Status theory weight (α parameter)')
    parser.add_argument('--csgu_expansion_depth', type=int, default=1, help='Triangle expansion depth (p parameter)')
    parser.add_argument('--csgu_cg_iterations', type=int, help='Conjugate gradient iterations (T parameter)')
    parser.add_argument('--csgu_damping', type=float, help='CG damping parameter (λ parameter)')
    parser.add_argument('--csgu_hessian_scale', type=float, help='Hessian scaling parameter (σ parameter)')
    parser.add_argument('--csgu_update_scale', type=float, help='Parameter update scaling factor (η parameter)')
    parser.add_argument('--csgu_epsilon', type=float, help='Privacy budget (ε parameter)')
    parser.add_argument('--csgu_delta', type=float, default=1e-5, help='Failure probability (δ parameter)')
    parser.add_argument('--csgu_clip_threshold', type=float, help='Gradient clipping threshold (τ parameter)')
    
    # IDEA specific parameters
    parser.add_argument('--idea_iteration', type=int, default=100, help='Number of iterations for IDEA approximation')
    parser.add_argument('--idea_damp', type=float, default=0.01, help='Damping factor for IDEA')
    parser.add_argument('--idea_scale', type=float, help='Scaling factor for IDEA')
    parser.add_argument('--idea_gaussian_std', type=float, default=0.01, help='Standard deviation for Gaussian noise in IDEA')
    parser.add_argument('--idea_gaussian_mean', type=float, help='Mean for Gaussian noise in IDEA')
    parser.add_argument('--idea_l', type=float, default=1.0, help='Lipschitz constant for IDEA certification')
    parser.add_argument('--idea_lambda', type=float, default=0.01, help='Strong convexity parameter for IDEA')
    parser.add_argument('--idea_c', type=float, default=1.0, help='Loss bound for IDEA certification')

    return parser.parse_args() 