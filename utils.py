from sys import argv

import torch
from torch_geometric_signed_directed.nn.signed import SDGNN, SGCN, SNEA, SiGAT


def get_model(model_name, nodes_num, edge_index_s, args, device):
    if model_name == 'SGCN':
        model = SGCN(node_num=nodes_num, edge_index_s=edge_index_s, in_dim=args.in_dim, out_dim=args.out_dim, layer_num=args.layer_num, lamb=5).to(device)
    elif model_name == 'SiGAT':
        model = SiGAT(node_num=nodes_num, 
                      edge_index_s=edge_index_s,
                      in_dim=args.in_dim, 
                      out_dim=args.out_dim).to(device)
    elif model_name == 'SNEA':
        edge_index = edge_index_s[:, :2].t()
        edge_weight = torch.ones(edge_index.size(1), device=device)
        model = SNEA(node_num=nodes_num, edge_index_s=edge_index_s, in_dim=args.in_dim, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    elif model_name == 'SDGNN':
        edge_index = edge_index_s[:, :2].t()
        edge_weight = torch.ones(edge_index.size(1), device=device)
        model = SDGNN(node_num=nodes_num, edge_index_s=edge_index_s, in_dim=args.in_dim, out_dim=args.out_dim, layer_num=args.layer_num).to(device)
    else:
        raise Exception('unsupported model')
    
    # Initialize model parameters
    if model_name != 'SGCN':
        model.reset_parameters()
    
    return model


def find_k_hops(start_nodes, k, edge_index):
    """
    Finds all nodes within k hops (considering the graph as undirected) from a set of start nodes.

    Args:
        start_nodes (torch.Tensor): A tensor of starting node indices.
        k (int): The number of hops.
        edge_index (torch.Tensor): The edge index of the graph (should be undirected).

    Returns:
        torch.Tensor: A tensor containing the unique node indices within k hops.
    """
    if k < 0:
        return torch.tensor([], dtype=torch.long, device=start_nodes.device)
    
    if start_nodes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=start_nodes.device)

    # Convert to a set for efficient operations
    k_hop_nodes = set(start_nodes.cpu().numpy())
    frontier = start_nodes.clone()
    
    for _ in range(k):
        if frontier.numel() == 0:
            break
            
        # Find neighbors of the current frontier
        # In an undirected graph, an edge (u, v) means v is a neighbor of u and u is a neighbor of v.
        is_in_frontier_src = torch.isin(edge_index[0], frontier)
        is_in_frontier_dst = torch.isin(edge_index[1], frontier)
        
        neighbors_from_src = edge_index[1, is_in_frontier_src]
        neighbors_from_dst = edge_index[0, is_in_frontier_dst]
        
        all_neighbors = torch.cat([neighbors_from_src, neighbors_from_dst]).unique()
        
        new_nodes_mask = ~torch.isin(all_neighbors, torch.tensor(list(k_hop_nodes), device=all_neighbors.device))
        new_frontier = all_neighbors[new_nodes_mask]
        
        k_hop_nodes.update(new_frontier.cpu().numpy())
        frontier = new_frontier

    return torch.tensor(list(k_hop_nodes), device=start_nodes.device)

