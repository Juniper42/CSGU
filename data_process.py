import argparse
import os
import os.path as osp

import pandas as pd
import torch
from torch_geometric_signed_directed.data.signed import load_signed_real_data


def train_test_gen(data, test_ratio=0.1, val_ratio=0.1, seed=2025):
    torch.manual_seed(seed)
    edge_index = data.edge_index
    edge_weight = data.edge_weight

    edge_dict = {(u.item(), v.item()): w.item() for u, v, w in zip(edge_index[0], edge_index[1], edge_weight)}

    num_edges = edge_index.shape[1]
    test_size = int(num_edges * test_ratio)

    perm = torch.randperm(num_edges)
    test_edge_indices = perm[:test_size]

    test_edge = edge_index[:, test_edge_indices].T
    test_labels = torch.tensor(
        [1 if edge_dict.get((u.item(), v.item()), edge_dict.get((v.item(), u.item()), 0)) > 0 else -1 for u, v in test_edge],
        dtype=torch.long
    )

    test_data = {"edges": test_edge, "label": test_labels}

    train_val_mask = torch.ones(num_edges, dtype=torch.bool)
    train_val_mask[test_edge_indices] = False
    train_val_edge_index = edge_index[:, train_val_mask].T

    train_val_labels = torch.tensor(
        [1 if edge_dict.get((u.item(), v.item()), edge_dict.get((v.item(), u.item()), 0)) > 0 else -1 for u, v in train_val_edge_index],
        dtype=torch.long
    )

    num_train_val = train_val_edge_index.shape[0]
    val_size = int(num_train_val * val_ratio)

    perm = torch.randperm(num_train_val)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    val_edge = train_val_edge_index[val_indices]
    val_labels = train_val_labels[val_indices]
    train_edge = train_val_edge_index[train_indices]
    train_labels = train_val_labels[train_indices]

    train_data = {"edges": train_edge, "label": train_labels}
    val_data = {"edges": val_edge, "label": val_labels}

    return train_data, val_data, test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    data = load_signed_real_data(dataset=args.dataset, root=path)

    train_data, val_data, test_data = train_test_gen(data, args.test_ratio, args.val_ratio, args.seed)

    save_dir = osp.join(path, args.dataset, 'processed', str(args.seed))
    os.makedirs(save_dir, exist_ok=True)

    for data_split, name, ratio in [(train_data, f'train_{1 - args.test_ratio - args.val_ratio}'), (val_data, f'val_{args.val_ratio}'), (test_data, f'test_{args.test_ratio}')]:
        df = pd.DataFrame({
            'source': data_split['edges'][:, 0].cpu().numpy(),
            'target': data_split['edges'][:, 1].cpu().numpy(),
            'label': data_split['label'].cpu().numpy()
        })
        file_path = osp.join(save_dir, f'{name}_{ratio}.parquet')
        df.to_parquet(file_path)


if __name__ == "__main__":
    main() 