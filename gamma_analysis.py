"""
Impact of Certification Region Size |R| ≤ γ|E| on CSGU Unlearning Performance

Examines how varying γ affects unlearning quality (Macro F1), privacy (MIA AUC),
and efficiency (Time) under SDGNN with 2.5% edge deletion.

Usage:
    python gamma_analysis.py                             # default γ sweep
    python gamma_analysis.py --device cuda               # force GPU
    python gamma_analysis.py --gammas 0.01 0.05 0.10 0.20 0.30
"""

import os
import os.path as osp
import copy
import time
import argparse
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd
from tabulate import tabulate

from torch_geometric_signed_directed.data.signed import load_signed_real_data
from torch_geometric.utils import degree

from data_process import train_test_gen
from train import train
from evaluate import evaluate, mia
from utils import get_model
from unlearning_func.CSGU import CSGU


GAMMAS = [0.01, 0.10, 0.20, 0.30]
DATASETS = ['bitcoin_alpha', 'slashdot']
DISPLAY = {'bitcoin_alpha': 'Alpha', 'slashdot': 'Slashdot'}


def make_args(dataset='bitcoin_alpha'):
    """Default SDGNN hyperparameters, 2.5% edge deletion."""
    return SimpleNamespace(
        dataset=dataset,
        model='SDGNN',
        in_dim=20, out_dim=20, layer_num=2,
        lr=0.01, weight_decay=5e-4,
        epochs=500, eval_step=5, patience=10,
        seed=42,
        test_ratio=0.1, val_ratio=0.1,
        runs=1, device='auto',
        node_feature_dim=20,
        unlearning_method='CSGU',
        unlearning_task='edge',
        unlearning_ratio=0.025,
        csgu_alpha=0.5, csgu_expansion_depth=1,
        csgu_cg_iterations=20, csgu_damping=0.1,
        csgu_hessian_scale=1.0, csgu_update_scale=0.1,
        csgu_epsilon=1.0, csgu_delta=1e-5,
        csgu_clip_threshold=1.0,
    )


# ---------------------------------------------------------------------------
#  Data loading  (mirrors main.py / tightness_analysis.py)
# ---------------------------------------------------------------------------

def generate_node_features(num_nodes, edge_index, feature_dim, device='cpu'):
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    base = torch.log(deg + 1).unsqueeze(1).repeat(1, feature_dim)
    noise = torch.randn(num_nodes, feature_dim, device=device) * 0.2
    return torch.nn.functional.normalize(base + noise, p=2, dim=1)


def load_data(args, device):
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset_path = osp.join(data_path, args.dataset, 'processed', str(args.seed))

    try:
        find = lambda prefix: next(
            f for f in os.listdir(dataset_path)
            if f.startswith(prefix) and f.endswith('.parquet')
        )
        dfs = {
            s: pd.read_parquet(osp.join(dataset_path, find(s)))
            for s in ('train', 'val', 'test')
        }
        splits = {
            s: {
                'edges': torch.from_numpy(df[['source', 'target']].values),
                'label': torch.from_numpy(df['label'].values),
            }
            for s, df in dfs.items()
        }
    except (FileNotFoundError, StopIteration, OSError):
        print("  No cached parquets; generating from raw dataset...")
        data = load_signed_real_data(dataset=args.dataset, root=data_path).to(device)
        td, vd, tsd = train_test_gen(data, args.test_ratio, args.val_ratio, args.seed)
        splits = {'train': td, 'val': vd, 'test': tsd}
        save_dir = osp.join(data_path, args.dataset, 'processed', str(args.seed))
        os.makedirs(save_dir, exist_ok=True)
        for sp_data, name, ratio in [
            (td, 'train', 1 - args.test_ratio - args.val_ratio),
            (vd, 'val', args.val_ratio),
            (tsd, 'test', args.test_ratio),
        ]:
            df = pd.DataFrame({
                'source': sp_data['edges'][:, 0].cpu().numpy(),
                'target': sp_data['edges'][:, 1].cpu().numpy(),
                'label': sp_data['label'].cpu().numpy(),
            })
            df.to_parquet(osp.join(save_dir, f'{name}_{ratio}.parquet'))

    for d in splits.values():
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device)

    return splits['train'], splits['val'], splits['test']


# ---------------------------------------------------------------------------
#  γ-constrained CSGU unlearning
# ---------------------------------------------------------------------------

def _cap_influence_region(csgu, influence_indices, target_edges, gamma, device):
    r"""
    Enforce |R| ≤ γ|E|.

    Priority: target (unlearned) edges are always retained so that the gradient
    difference ∇L_w(R) − ∇L_w(R \ D_u) remains meaningful; remaining capacity
    is filled by uniform sampling from the non-target influence edges.
    """
    total_E = csgu.train_edges.size(0)
    max_R = int(gamma * total_E)

    if influence_indices.size(0) <= max_R:
        return influence_indices

    target_idx_set = set()
    for e in target_edges.cpu().numpy():
        idx = csgu.edge_to_idx.get((int(e[0]), int(e[1])))
        if idx is None:
            idx = csgu.edge_to_idx.get((int(e[1]), int(e[0])))
        if idx is not None:
            target_idx_set.add(idx)

    inf_arr = influence_indices.cpu().numpy()
    is_target = np.isin(inf_arr, list(target_idx_set))

    target_part = inf_arr[is_target]
    non_target_part = inf_arr[~is_target]

    budget = max(0, max_R - len(target_part))
    if len(non_target_part) > budget:
        chosen = np.random.choice(non_target_part, budget, replace=False)
    else:
        chosen = non_target_part

    capped = np.concatenate([target_part, chosen])
    return torch.tensor(capped, dtype=torch.long, device=device)


def csgu_unlearn_with_gamma(csgu, original_model, unlearned_data, gamma, device):
    r"""
    Full CSGU unlearning pipeline with the constraint |R| ≤ γ|E|.

    Steps mirror CSGU.unlearn() but insert a cap after the triangle expansion:
      1. Triangle-based influence region → apply γ cap
      2. Sociological influence quantification
      3. Weighted gradient difference
      4. Certified conjugate gradient + Gaussian noise
      5. Clip & update parameters

    Returns (elapsed_seconds, unlearned_model, actual_R_size).
    """
    model = copy.deepcopy(original_model)
    model.to(device)
    model.eval()

    t0 = time.time()

    # Step 1 — influence region + γ cap
    target_edges = csgu._determine_target_edges(unlearned_data)
    influence_full = csgu._triangle_based_influence_region(target_edges)
    influence_indices = _cap_influence_region(
        csgu, influence_full, target_edges, gamma, device
    )

    max_R = int(gamma * csgu.train_edges.size(0))
    print(f"    |R_full|={influence_full.size(0)}, "
          f"|R_cap|={influence_indices.size(0)}, "
          f"γ·|E|={max_R}")

    if influence_indices.size(0) == 0:
        return time.time() - t0, model, 0

    influence_edges = csgu.train_edges[influence_indices]
    influence_nodes = torch.unique(influence_edges.flatten())

    if influence_nodes.size(0) == 0:
        return time.time() - t0, model, 0

    # Step 2 — sociological weights
    node_importance = csgu._compute_unified_centrality_and_influence(influence_nodes)

    # Step 3 — gradient difference
    gradient_diff = csgu._compute_gradient_difference(
        model, influence_indices, unlearned_data, node_importance
    )

    # Step 4 — certified CG + noise
    param_changes = csgu._certified_conjugate_gradient(
        model, gradient_diff, unlearned_data
    )

    # Step 5 — clip & update
    csgu._clip_and_update_parameters(model, param_changes)

    elapsed = time.time() - t0
    return elapsed, model, int(influence_indices.size(0))


# ---------------------------------------------------------------------------
#  Per-dataset experiment
# ---------------------------------------------------------------------------

def run_dataset(dataset_name, device, gammas):
    display = DISPLAY.get(dataset_name, dataset_name)
    print(f"\n{'=' * 60}")
    print(f" Dataset: {display}")
    print(f"{'=' * 60}")

    args = make_args(dataset_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data ----
    print("[1/4] Loading data...")
    train_data, val_data, test_data = load_data(args, device)
    link_data = {'train': train_data, 'val': val_data, 'test': test_data}

    data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    data_obj = load_signed_real_data(
        dataset=args.dataset, root=data_path
    ).to(device)
    nodes_num = data_obj.num_nodes

    edge_index = torch.cat([train_data['edges'], val_data['edges']], dim=0)
    edge_sign = torch.cat([train_data['label'], val_data['label']], dim=0)
    edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)

    # ---- model (trained once, shared across γ) ----
    print("[2/4] Preparing SDGNN model...")
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
        print("  Training model...")
        model = train(model, optimizer, link_data, args, device)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    model.eval()

    # ---- unlearned edges (2.5%, fixed across γ) ----
    print("[3/4] Sampling 2.5% edges for deletion...")
    num_train = train_data['edges'].size(0)
    unlearn_size = int(num_train * args.unlearning_ratio)
    perm = torch.randperm(num_train, device=device)
    unlearn_idx = perm[:unlearn_size]

    unlearned_edges = train_data['edges'][unlearn_idx]
    unlearned_labels = train_data['label'][unlearn_idx]

    keep = torch.ones(num_train, dtype=torch.bool, device=device)
    keep[unlearn_idx] = False
    remaining_edges = train_data['edges'][keep]
    remaining_labels = train_data['label'][keep]

    remaining_link_data = {
        'train': {'edges': remaining_edges, 'label': remaining_labels},
        'val': val_data, 'test': test_data,
    }
    unlearned_data = {
        'unlearn_indices': unlearn_idx,
        'unlearned_edges': unlearned_edges,
        'unlearned_labels': unlearned_labels,
        'unlearned_nodes_num': torch.unique(unlearned_edges.flatten()).size(0),
        'unlearned_edge_index_s': torch.cat(
            [unlearned_edges, unlearned_labels.unsqueeze(-1)], dim=-1
        ),
        'nodes_num': nodes_num,
        'remaining_link_data': remaining_link_data,
    }

    all_edges = torch.cat([
        train_data['edges'], val_data['edges'], test_data['edges']
    ], dim=0)

    print(f"  Train edges: {num_train}, Unlearned: {unlearn_size}")

    # ---- CSGU instance (shared) ----
    csgu = CSGU(link_data, args, device)

    # ---- sweep γ ----
    print(f"[4/4] Running γ sweep: {gammas}")
    results = []

    for gamma in gammas:
        print(f"\n  γ = {gamma:.2f}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        elapsed, unlearned_model, R_size = csgu_unlearn_with_gamma(
            csgu, model, unlearned_data, gamma, device
        )

        eval_info = evaluate(
            unlearned_model, remaining_link_data, device, eval_flag='test'
        )
        mia_info = mia(unlearned_model, unlearned_data, all_edges, device)

        f1 = eval_info['f1_macro'] * 100
        mia_auc = mia_info['mia_auc'] * 100

        print(f"    F1={f1:.2f}  MIA_AUC={mia_auc:.2f}  "
              f"Time={elapsed:.1f}s  |R|={R_size}")

        results.append({
            'gamma': gamma,
            'f1': f1,
            'mia_auc': mia_auc,
            'time': round(elapsed, 1),
            'R_size': R_size,
        })

    return display, results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Impact of |R| ≤ γ|E| on CSGU unlearning performance'
    )
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--gammas', type=float, nargs='+', default=GAMMAS,
        help='γ values to sweep (default: 0.01 0.10 0.20 0.30)',
    )
    parser.add_argument(
        '--datasets', type=str, nargs='+', default=DATASETS,
        help='Datasets to evaluate (default: bitcoin_alpha slashdot)',
    )
    cli = parser.parse_args()

    if cli.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cli.device)
    print(f"Device: {device}\n")

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    # ---- run ----
    dataset_results = {}
    for ds in cli.datasets:
        try:
            display, results = run_dataset(ds, device, cli.gammas)
            dataset_results[display] = {r['gamma']: r for r in results}
        except Exception as e:
            print(f"\n  ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    if not dataset_results:
        print("No results collected.")
        return

    # ---- summary table ----
    ds_names = [DISPLAY.get(d, d) for d in cli.datasets if DISPLAY.get(d, d) in dataset_results]

    print(f"\n{'=' * 80}")
    print("Impact of |R| ≤ γ|E| on SDGNN  (2.5% edge deletion)")
    print("Metric format: Macro-F1↑ / MIA AUC↓ / Time(s)↓")
    print(f"{'=' * 80}\n")

    headers = ['γ'] + [f'{n}:F1↑/MIAUC↓/Time(s)↓' for n in ds_names]
    rows = []
    for gamma in cli.gammas:
        row = [f'{gamma:.2f}']
        for n in ds_names:
            r = dataset_results.get(n, {}).get(gamma)
            if r:
                row.append(f"{r['f1']:.2f}/{r['mia_auc']:.2f}/{r['time']}")
            else:
                row.append('-')
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt='pipe'))

    # ---- persist ----
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    flat = []
    for ds_disp, by_gamma in dataset_results.items():
        for gamma, r in sorted(by_gamma.items()):
            flat.append({'dataset': ds_disp, **r})

    out = osp.join(results_dir, 'gamma_analysis.csv')
    pd.DataFrame(flat).to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
