import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric_signed_directed.utils.signed import \
    link_sign_prediction_logistic_function


def evaluate(model, link_data, device, eval_flag='test'):
    model.eval()
    with torch.no_grad():
        z = model()
    embeddings = z
    train_X = link_data['train']['edges']
    test_X = link_data[eval_flag]['edges']
    train_y = link_data['train']['label']
    test_y = link_data[eval_flag]['label']

    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(
        embeddings.cpu().numpy(), train_X.cpu().numpy(), train_y.cpu().numpy(), test_X.cpu().numpy(), test_y.cpu().numpy())
    eval_info = {}
    eval_info['acc'] = accuracy
    eval_info['f1'] = f1
    eval_info['f1_macro'] = f1_macro
    eval_info['f1_micro'] = f1_micro
    eval_info['auc'] = auc_score
    return eval_info

def mia(model, unlearned_data, all_edges, device):
    """
    Performs Membership Inference Attack (MIA) evaluation tailored for signed graphs.

    Args:
        model: The model to evaluate.
        unlearned_data (dict): A dictionary containing the data that was unlearned.
                               It must contain 'unlearned_edges' and 'unlearned_labels'.
        all_edges (torch.Tensor): A tensor containing all edges in the graph (train, val, test)
                                  to ensure sampling of true non-member edges.
        device: The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the MIA AUC score.
    """
    model.eval()
    with torch.no_grad():
        z = model()
    embeddings = z.cpu().numpy()
    num_nodes = embeddings.shape[0]

    # Get the unlearned edges (members) and their labels
    member_edges = unlearned_data['unlearned_edges'].cpu().numpy()
    if 'unlearned_labels' not in unlearned_data:
        raise ValueError("unlearned_data must contain 'unlearned_labels' for signed MIA.")
    
    # Generate true non-member edges
    existing_edges = set()
    for u, v in all_edges.cpu().numpy():
        existing_edges.add(tuple(sorted((u,v))))

    non_member_edges = []
    num_non_members = len(member_edges)
    while len(non_member_edges) < num_non_members:
        u, v = np.random.randint(0, num_nodes, size=2)
        if u != v and tuple(sorted((u,v))) not in existing_edges:
            non_member_edges.append((u, v))
    non_member_edges = np.array(non_member_edges)

    # Predict scores for members and non-members
    member_scores = np.sum(embeddings[member_edges[:, 0]] * embeddings[member_edges[:, 1]], axis=1)
    non_member_scores = np.sum(embeddings[non_member_edges[:, 0]] * embeddings[non_member_edges[:, 1]], axis=1)
    
    # For MIA, we care about the model's confidence. High absolute scores for members (positive or negative)
    # vs low absolute scores for non-members (scores around 0).
    scores = np.concatenate([np.abs(member_scores), np.abs(non_member_scores)])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
    
    auc_score = roc_auc_score(labels, scores)
    
    return {'mia_auc': auc_score}
    