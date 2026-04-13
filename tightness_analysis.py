"""
Tightness Analysis of Gradient Norm Upper Bound for CSGU Certified Removal

Validates (ε,δ)-certified deletion effectiveness by computing the tightness ratio
between actual per-edge gradient norms and the theoretical upper bound (Eq. 12)
for all edges in the certification region R.

Setup: SDGNN model, 2.5% edge deletion, five signed graph datasets.

Usage:
    python tightness_analysis.py                          # all datasets
    python tightness_analysis.py --dataset bitcoin_alpha  # single dataset
    python tightness_analysis.py --device cuda             # force GPU
"""

import os
import os.path as osp
import math
import time
import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import pandas as pd
from tabulate import tabulate

from torch_geometric_signed_directed.data.signed import load_signed_real_data
from torch_geometric.utils import degree

from data_process import train_test_gen
from train import train
from evaluate import evaluate
from utils import get_model
from unlearning_func.CSGU import CSGU


DATASETS = ['bitcoin_alpha', 'bitcoin_otc', 'wiki', 'epinions', 'slashdot']
DISPLAY_NAMES = {
    'bitcoin_alpha': 'Bitcoin-Alpha',
    'bitcoin_otc': 'Bitcoin-OTC',
    'wiki': 'WikiRfA',
    'epinions': 'Epinions',
    'slashdot': 'Slashdot',
}


def make_args(dataset='bitcoin_alpha'):
    """Create args namespace with default hyperparameters for SDGNN."""
    return SimpleNamespace(
        dataset=dataset,
        model='SDGNN',
        in_dim=20,
        out_dim=20,
        layer_num=2,
        lr=0.01,
        weight_decay=5e-4,
        epochs=500,
        eval_step=5,
        patience=10,
        seed=42,
        test_ratio=0.1,
        val_ratio=0.1,
        runs=1,
        device='auto',
        node_feature_dim=20,
        unlearning_method='CSGU',
        unlearning_task='edge',
        unlearning_ratio=0.025,
        csgu_alpha=0.5,
        csgu_expansion_depth=1,
        csgu_cg_iterations=20,
        csgu_damping=0.1,
        csgu_hessian_scale=1.0,
        csgu_update_scale=0.1,
        csgu_epsilon=1.0,
        csgu_delta=1e-5,
        csgu_clip_threshold=1.0,
    )


def generate_node_features(num_nodes, edge_index, feature_dim, device='cpu'):
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    base_features = torch.log(deg + 1).unsqueeze(1).repeat(1, feature_dim)
    noise = torch.randn(num_nodes, feature_dim, device=device) * 0.2
    node_features = base_features + noise
    return torch.nn.functional.normalize(node_features, p=2, dim=1)


def load_data(args, device):
    """Load signed graph data following the main.py pipeline."""
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset_path = osp.join(data_path, args.dataset, 'processed', str(args.seed))

    try:
        train_path = [f for f in os.listdir(dataset_path)
                      if f.startswith('train_') and f.endswith('.parquet')][0]
        val_path = [f for f in os.listdir(dataset_path)
                    if f.startswith('val_') and f.endswith('.parquet')][0]
        test_path = [f for f in os.listdir(dataset_path)
                     if f.startswith('test_') and f.endswith('.parquet')][0]

        train_df = pd.read_parquet(osp.join(dataset_path, train_path))
        val_df = pd.read_parquet(osp.join(dataset_path, val_path))
        test_df = pd.read_parquet(osp.join(dataset_path, test_path))

        train_data = {
            'edges': torch.from_numpy(train_df[['source', 'target']].values),
            'label': torch.from_numpy(train_df['label'].values),
        }
        val_data = {
            'edges': torch.from_numpy(val_df[['source', 'target']].values),
            'label': torch.from_numpy(val_df['label'].values),
        }
        test_data = {
            'edges': torch.from_numpy(test_df[['source', 'target']].values),
            'label': torch.from_numpy(test_df['label'].values),
        }
    except (FileNotFoundError, IndexError, OSError):
        print("  Preprocessed data not found, generating from raw dataset...")
        data = load_signed_real_data(dataset=args.dataset, root=data_path).to(device)
        train_data, val_data, test_data = train_test_gen(
            data, args.test_ratio, args.val_ratio, args.seed
        )
        save_dir = osp.join(data_path, args.dataset, 'processed', str(args.seed))
        os.makedirs(save_dir, exist_ok=True)
        for split, name, ratio in [
            (train_data, 'train', 1 - args.test_ratio - args.val_ratio),
            (val_data, 'val', args.val_ratio),
            (test_data, 'test', args.test_ratio),
        ]:
            df = pd.DataFrame({
                'source': split['edges'][:, 0].cpu().numpy(),
                'target': split['edges'][:, 1].cpu().numpy(),
                'label': split['label'].cpu().numpy(),
            })
            df.to_parquet(osp.join(save_dir, f'{name}_{ratio}.parquet'))

    for d in [train_data, val_data, test_data]:
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device)

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
#  Theoretical upper bound (Eq. 12)
# ---------------------------------------------------------------------------

def _spectral_norm_product(model):
    """Product of spectral norms (σ_max) of all weight matrices in the model."""
    prod = 1.0
    for _, param in model.named_parameters():
        if param.dim() >= 2:
            try:
                s = torch.linalg.svdvals(param.detach())
                prod *= s[0].item()
            except Exception:
                prod *= param.detach().norm().item()
    return prod


def compute_theoretical_bound(model, influence_edges, node_weights, num_nodes, device):
    r"""
    Theoretical per-edge gradient norm upper bound B (Eq. 12).

    For weighted BCE loss on edge (u,v) with prediction s = z_u^T z_v:

        ||∇_θ ℓ_w(e; θ)|| = w_{uv} · |σ(s)-y| · ||∇_θ s||

    Worst-case bounds:
        • |σ(s) - y| ≤ 1            (BCE output range)
        • ||∇_θ s||  ≤ 2 · M_z · J  (chain rule, both endpoints)

    where
        M_z = max_v ||z_v||_2         (maximum embedding norm)
        J   = M_z · Π_l σ_1(W_l)     (Jacobian spectral bound via GNN layers)

    Global bound:
        B = w_max · 2 · M_z² · Π_l σ_1(W_l) · L

    L is the number of GNN layers (accounts for multi-layer Jacobian).
    """
    model.eval()
    with torch.no_grad():
        z = model()

    max_emb_norm = z.norm(dim=1).max().item()

    w_max = 0.0
    for i in range(influence_edges.size(0)):
        u, v = influence_edges[i, 0].item(), influence_edges[i, 1].item()
        w_u = node_weights.get(u, 0.0)
        w_v = node_weights.get(v, 0.0)
        w_max = max(w_max, (w_u + w_v) * 0.5)

    spec_prod = _spectral_norm_product(model)
    num_layers = sum(1 for _, p in model.named_parameters() if p.dim() >= 2)
    num_layers = max(num_layers // 2, 1)

    B = w_max * 2.0 * (max_emb_norm ** 2) * spec_prod * num_layers
    return B


# ---------------------------------------------------------------------------
#  Per-edge actual gradient norms
# ---------------------------------------------------------------------------

def compute_per_edge_gradient_norms(
    model, edges, labels, node_weights, num_nodes, device,
    max_edges=None,
):
    """
    Compute ||∇_θ ℓ_w(e; θ)||_2 for each edge e in the influence region.

    If max_edges is set and |edges| exceeds it, a uniform random subsample
    is taken for tractability on large graphs.
    """
    n_edges = edges.size(0)
    if max_edges and n_edges > max_edges:
        idx = torch.randperm(n_edges, device=device)[:max_edges]
        edges = edges[idx]
        labels = labels[idx]
        n_edges = max_edges
        print(f"  Subsampled to {n_edges} edges for gradient computation")

    model_params = [p for p in model.parameters() if p.requires_grad]

    node_w = torch.zeros(num_nodes, device=device)
    if node_weights:
        wk = torch.tensor(list(node_weights.keys()), device=device, dtype=torch.long)
        wv = torch.tensor(list(node_weights.values()), device=device)
        node_w[wk] = wv

    norms = np.empty(n_edges, dtype=np.float64)

    t0 = time.time()
    for i in range(n_edges):
        z = model()
        u, v = edges[i, 0], edges[i, 1]
        logit = (z[u] * z[v]).sum().unsqueeze(0)
        w_uv = ((node_w[u] + node_w[v]) * 0.5).clamp(min=1e-12).unsqueeze(0)

        loss = F.binary_cross_entropy_with_logits(
            logit, labels[i].float().unsqueeze(0),
            weight=w_uv, reduction='sum',
        )

        g = grad(loss, model_params, retain_graph=False, allow_unused=True)
        norm_sq = sum(
            (gi.norm() ** 2).item()
            for gi, p in zip(g, model_params)
            if gi is not None
        )
        norms[i] = math.sqrt(norm_sq)

        if (i + 1) % 500 == 0 or i + 1 == n_edges:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_edges - i - 1)
            print(f"  [{i+1}/{n_edges}] elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

    return norms


# ---------------------------------------------------------------------------
#  Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(dataset_name, device, max_edges=2000):
    """Full tightness analysis for one dataset."""
    display = DISPLAY_NAMES.get(dataset_name, dataset_name)
    print(f"\n{'=' * 60}")
    print(f" Dataset: {display}")
    print(f"{'=' * 60}")

    args = make_args(dataset_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- data ---
    print("[1/6] Loading data...")
    train_data, val_data, test_data = load_data(args, device)
    link_data = {'train': train_data, 'val': val_data, 'test': test_data}

    data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    data_obj = load_signed_real_data(dataset=args.dataset, root=data_path).to(device)
    nodes_num = data_obj.num_nodes

    edge_index = torch.cat([train_data['edges'], val_data['edges']], dim=0)
    edge_sign = torch.cat([train_data['label'], val_data['label']], dim=0)
    edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)

    print(f"  Nodes: {nodes_num}, Train edges: {train_data['edges'].size(0)}")

    # --- model ---
    print("[2/6] Preparing SDGNN model...")
    model = get_model('SDGNN', nodes_num, edge_index_s, args, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    cache_dir = ".cache"
    tr = 1 - args.test_ratio - args.val_ratio
    fname = (f"SDGNN_i{args.in_dim}_o{args.out_dim}_layer{args.layer_num}"
             f"_lr{args.lr}_wd{args.weight_decay}_{args.dataset}"
             f"_seed{args.seed}_train{tr}.pt")
    model_path = osp.join(cache_dir, fname)

    if osp.exists(model_path):
        print(f"  Loaded cached model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("  Training model (no cache found)...")
        model = train(model, optimizer, link_data, args, device)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"  Model cached to {model_path}")

    model.eval()

    # --- unlearned edges (2.5%) ---
    print("[3/6] Sampling 2.5% edges for deletion...")
    num_train = train_data['edges'].size(0)
    unlearn_size = int(num_train * args.unlearning_ratio)
    perm = torch.randperm(num_train, device=device)
    unlearn_idx = perm[:unlearn_size]

    unlearned_edges = train_data['edges'][unlearn_idx]
    unlearned_labels = train_data['label'][unlearn_idx]

    keep_mask = torch.ones(num_train, dtype=torch.bool, device=device)
    keep_mask[unlearn_idx] = False
    remaining_edges = train_data['edges'][keep_mask]
    remaining_labels = train_data['label'][keep_mask]

    unlearned_data = {
        'unlearn_indices': unlearn_idx,
        'unlearned_edges': unlearned_edges,
        'unlearned_labels': unlearned_labels,
        'unlearned_nodes_num': torch.unique(unlearned_edges.flatten()).size(0),
        'unlearned_edge_index_s': torch.cat(
            [unlearned_edges, unlearned_labels.unsqueeze(-1)], dim=-1
        ),
        'nodes_num': nodes_num,
        'remaining_link_data': {
            'train': {'edges': remaining_edges, 'label': remaining_labels},
            'val': val_data, 'test': test_data,
        },
    }
    print(f"  Unlearned edges: {unlearn_size}")

    # --- certification region R ---
    print("[4/6] Computing certification region R (triangle expansion)...")
    csgu = CSGU(link_data, args, device)
    target_edges = csgu._determine_target_edges(unlearned_data)
    influence_indices = csgu._triangle_based_influence_region(target_edges)

    if influence_indices.size(0) == 0:
        print("  WARNING: empty influence region — skipping dataset.")
        return None

    influence_edges = csgu.train_edges[influence_indices]
    influence_labels = csgu.train_labels[influence_indices]
    influence_nodes = torch.unique(influence_edges.flatten())

    print(f"  |R| = {influence_indices.size(0)} edges, "
          f"{influence_nodes.size(0)} nodes")

    # --- sociological weights ---
    node_weights = csgu._compute_unified_centrality_and_influence(influence_nodes)

    # --- per-edge gradient norms ---
    print(f"[5/6] Computing per-edge gradient norms (max {max_edges})...")
    actual_norms = compute_per_edge_gradient_norms(
        model, influence_edges, influence_labels,
        node_weights, csgu.num_nodes, device,
        max_edges=max_edges,
    )

    # --- theoretical bound ---
    print("[6/6] Computing theoretical gradient norm upper bound (Eq. 12)...")
    B = compute_theoretical_bound(
        model, influence_edges, node_weights, csgu.num_nodes, device
    )
    print(f"  B = {B:.6f}")

    # --- tightness ratios ---
    ratios = actual_norms / B

    stats = {
        'dataset': display,
        'avg_ratio': float(np.mean(ratios)),
        'median':    float(np.median(ratios)),
        'p90':       float(np.percentile(ratios, 90)),
        'max_ratio': float(np.max(ratios)),
        'std':       float(np.std(ratios)),
        'n_edges_R': int(influence_indices.size(0)),
        'bound_B':   float(B),
    }

    print(f"\n  --- {display} ---")
    print(f"  Avg Ratio : {stats['avg_ratio']:.3f}")
    print(f"  Median    : {stats['median']:.3f}")
    print(f"  90th Pctl : {stats['p90']:.3f}")
    print(f"  Max       : {stats['max_ratio']:.3f}")
    print(f"  Std       : {stats['std']:.3f}")

    return stats


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Tightness analysis of gradient norm upper bound '
                    'for CSGU certified removal'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Single dataset name (e.g. bitcoin_alpha). '
             'Omit to run all five datasets.',
    )
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--max_edges', type=int, default=2000,
        help='Max edges to evaluate per dataset for tractability.',
    )
    cli = parser.parse_args()

    if cli.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cli.device)
    print(f"Device: {device}\n")

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    datasets = [cli.dataset] if cli.dataset else DATASETS
    all_stats = []

    for ds in datasets:
        try:
            stats = run_dataset(ds, device, max_edges=cli.max_edges)
            if stats is not None:
                all_stats.append(stats)
        except Exception as e:
            print(f"\n  ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    if not all_stats:
        print("No results collected.")
        return

    # ---- summary table ----
    print(f"\n{'=' * 70}")
    print("Tightness Ratio of Gradient Norm Upper Bound")
    print("(SDGNN, 2.5% edge deletion, certification region R)")
    print(f"{'=' * 70}\n")

    rows = [
        [
            s['dataset'],
            f"{s['avg_ratio']:.3f}",
            f"{s['median']:.3f}",
            f"{s['p90']:.3f}",
            f"{s['max_ratio']:.3f}",
            f"{s['std']:.3f}",
        ]
        for s in all_stats
    ]
    headers = ['Dataset', 'Avg Ratio', 'Median', '90th Pctl', 'Max', 'Std']
    print(tabulate(rows, headers=headers, tablefmt='pipe'))

    # ---- persist ----
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    out_path = osp.join(results_dir, 'tightness_analysis.csv')
    pd.DataFrame(all_stats).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
